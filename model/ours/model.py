import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from model.ours.nlq_head import NLQHead

from transformers.models.t5.modeling_t5 import T5Stack
from einops import rearrange

def interp_env_feat(tensor, B, T_target, mode='nearest'):
    D, dtype = tensor.shape[-1], tensor.dtype
    return F.interpolate(tensor[None, None].float(), size=(B, T_target, D), mode=mode).squeeze([0, 1]).to(dtype=dtype)

class GroundVQA(nn.Module):
    def __init__(self, lm_path, input_dim, ignore_decoder, vid_env_arch, query_env_arch, env_feature_dim, vid_sum_arch, freeze_word=False, max_v_len=256):
        super().__init__()
        self.ignore_decoder = ignore_decoder
        self.vid_env_arch = vid_env_arch
        self.query_env_arch = query_env_arch
        self.env_feature_dim = env_feature_dim
        self.vid_sum_arch = vid_sum_arch

        if not isinstance(input_dim, int):
            input_dim = input_dim.v_dim

        ########## for forward_encoder ##########
        self.lm: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(lm_path, local_files_only=True)
        if ignore_decoder:
            self.lm.decoder = None
            self.lm.lm_head = None

        lm_dim = self.lm.get_input_embeddings().embedding_dim
        self.lm_proj = nn.Linear(input_dim, lm_dim)
        self.v_emb = nn.Parameter(torch.randn((1, 1, lm_dim)))
        if freeze_word:
            for name, param in self.lm.named_parameters():
                if 'shared' in name:
                    param.requires_grad = False
        ########## for forward_encoder ##########
        
        ########## for after forward function ##########
        if self.vid_env_arch == 'concat':
            self.vid_env_proj = nn.Linear(self.env_feature_dim + lm_dim, lm_dim)
        
        if self.query_env_arch == 'ca':
            config = self.lm.config
            config.num_layers = 1
            config.is_decoder = True
            config.d_model = self.env_feature_dim
            
            self.env_q_sbert_attn = T5Stack(config, self.lm.get_input_embeddings())
            
            self.gamma = nn.Parameter(torch.randn(1))
            
        if self.vid_sum_arch == 'vid_t_proj':
            self.vid_t_proj = nn.Linear(1200, 600) 
            self.vid_sum_proj = nn.Linear(lm_dim * 2, lm_dim)
            
            
            
        ########## for after forward function ##########

        self.nlq_head = NLQHead(in_dim=lm_dim, max_v_len=max_v_len)

    def forward(self, v_feat, v_mask, 
                q_token, q_mask, 
                gt_segments, gt_labels, 
                q_sbert, q_sbert_mask, 
                env_feat, env_mask, labels=None, **remains):
        # encoder
        encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask) 
        
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]
        
        encoder_out_v = self.after_forward(encoder_out_v, q_sbert, q_sbert_mask, env_feat, env_mask, v_mask, v_feat)
        
        # localizer
        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            gt_segments=gt_segments,
            gt_labels=gt_labels
        )
        time_loss = nlq_results['final_loss'] * 1.0

        # decoder
        if self.ignore_decoder:
            return time_loss, 0, time_loss
        else:
            outputs = self.lm(
                encoder_outputs=(encoder_out,),
                attention_mask=mask,
                labels=labels,
            )
            lm_loss = outputs.loss
            total_loss = 0.5 * time_loss + 0.5 * lm_loss
            return total_loss, lm_loss, time_loss

    def generate(self, v_feat, v_mask, v_len,
                q_token, q_mask,
                q_sbert, q_sbert_mask, 
                env_feat, env_mask, **remains):
        encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask)
        
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]
        
        encoder_out_v = self.after_forward(encoder_out_v, q_sbert, q_sbert_mask, env_feat, env_mask, v_mask, v_feat)

        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            training=False,
            v_lens=v_len
        )
        
        if self.ignore_decoder:
            answer_tokens = None
        else:
            answer_tokens = self.lm.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
                attention_mask=mask,
                max_new_tokens=32
            )

        return nlq_results, answer_tokens

    def forward_encoder(self, v_feat, v_mask, q_token, q_mask):
        B, L, D = v_feat.shape
        v_feat = self.lm_proj(v_feat) # [B, T, 768]
        v_feat = v_feat + self.v_emb.expand((B, L, -1)) # [B, T, 768]
        q_feat = self.lm.encoder.embed_tokens(q_token)
        lm_input = torch.cat([q_feat, v_feat], dim=1) # [B, T + q, 768]
        lm_mask = torch.cat([q_mask, v_mask], dim=1) # [B, T + q, 768]
        out = self.lm.encoder(
            inputs_embeds=lm_input,
            attention_mask=lm_mask
        ) # [B, T + q, 768]
        return out.last_hidden_state, lm_mask
    
    def after_forward(self, encoder_out_v, q_sbert, q_sbert_mask, env_feat, env_mask, v_mask, v_feat):
        
        B, T, D = encoder_out_v.shape
        
        ########## query, env feat ##########
        if self.query_env_arch == 'ca':
            res_env = env_feat
                    
            q_sbert_mask = torch.ones((B, 1)).cuda()
            out = self.env_q_sbert_attn.forward(
                inputs_embeds=q_sbert, # q 독일어
                attention_mask=q_sbert_mask, # q
                encoder_hidden_states=env_feat, # k, v 출발어(영어)
                use_cache=False,
                return_dict=True
            ).last_hidden_state
            
            env_feat = res_env + F.tanh(self.gamma) * out
        ########## query, env feat ##########
        
        ########## vid summary ##########
        if self.vid_sum_arch == 'vid_t_proj':
            B, T, D = encoder_out_v.shape
        
            # Padding condition
            if v_feat.shape[1] != 1200:
                encoder_out_v_padded = F.pad(encoder_out_v, (0, 0, 0, 1200 - T), "constant", 0)  # [B, 1200, D]
            else:
                encoder_out_v_padded = encoder_out_v

            # Rearranging the tensor for projection
            encoder_out_v_padded = rearrange(encoder_out_v_padded, 'b t d -> b d t')  # [B, D, 1200]
            encoder_out_v_padded = rearrange(encoder_out_v_padded, 'b d t -> (b d) t')  # [B * D, 1200]
            
            # Applying the projection
            encoder_out_v_padded = self.vid_t_proj(encoder_out_v_padded)  # [B * D, proj_T]
            
            # Rearranging back to original dimensions
            encoder_out_v_padded = rearrange(encoder_out_v_padded, '(b d) t -> b t d', b=B, d=D)  # [B, proj_T, D]
            
            # Interpolating back to original shape
            encoder_out_v_padded = F.interpolate(encoder_out_v_padded.unsqueeze(0).unsqueeze(0).to(torch.float32), size=(B, T, D), mode='nearest').squeeze(0).squeeze(0)  # [B, T, D]

            # Applying the mask and combining with original
            encoder_out_v_padded = encoder_out_v_padded * v_mask.unsqueeze(2)
            
            res = encoder_out_v
            encoder_out_v = self.vid_sum_proj(torch.cat([encoder_out_v, encoder_out_v_padded], dim=2))
            encoder_out_v = F.gelu(encoder_out_v)
            encoder_out_v = res + encoder_out_v
        
        ########## vid, env feat ##########
        if self.vid_env_arch == 'concat':
            env_feat = interp_env_feat(env_feat, B, T, mode='nearest') # [B, T, D]
            
            res = encoder_out_v
            encoder_out_v = torch.cat([encoder_out_v, env_feat], dim=2)
            encoder_out_v = self.vid_env_proj(encoder_out_v)
            # encoder_out_v = torch.relu(encoder_out_v)
            encoder_out_v = F.gelu(encoder_out_v)
            
            encoder_out_v = res + encoder_out_v
        ########## vid, env feat ##########
            
        return encoder_out_v
