import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from model.ours.nlq_head import NLQHead

from transformers.models.t5.modeling_t5 import T5Stack

def interp_env_feat(tensor, B, T_target, mode='nearest'):
    D, dtype = tensor.shape[-1], tensor.dtype
    return F.interpolate(tensor[None, None].float(), size=(B, T_target, D), mode=mode).squeeze([0, 1]).to(dtype=dtype)

class GroundVQA(nn.Module):
    def __init__(self, lm_path, input_dim, ignore_decoder, vid_env_arch, query_env_arch, env_feature_dim, freeze_word=False, max_v_len=256):
        super().__init__()
        self.ignore_decoder = ignore_decoder
        self.vid_env_arch = vid_env_arch
        self.query_env_arch = query_env_arch
        self.env_feature_dim = env_feature_dim

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
        
        # if self.query_env_arch == 'ca':
        #     config = self.lm.config
        #     config.num_layers = 1
        #     config.is_decoder = True
        #     config.d_model = self.env_feature_dim
            
        #     self.env_q_sbert_attn = T5Stack(config, self.lm.get_input_embeddings())
            
        #     self.gamma = nn.Parameter(torch.randn(1))
            
            
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
        v_feat = self.lm_proj(v_feat)
        v_feat = v_feat + self.v_emb.expand((B, L, -1))
        q_feat = self.lm.encoder.embed_tokens(q_token)
        lm_input = torch.cat([q_feat, v_feat], dim=1)
        lm_mask = torch.cat([q_mask, v_mask], dim=1)
        out = self.lm.encoder(
            inputs_embeds=lm_input,
            attention_mask=lm_mask
        )
        return out.last_hidden_state, lm_mask
    
    def after_forward(self, encoder_out_v, q_sbert, q_sbert_mask, env_feat, env_mask, v_mask, v_feat):
        
        B, T, D = encoder_out_v.shape
        
        ########## query, env feat ##########
        # if self.query_env_arch == 'ca':
        #     res_env = env_feat
                    
        #     q_sbert_mask = torch.ones((B, 1)).cuda()
        #     out = self.env_q_sbert_attn.forward(
        #         inputs_embeds=q_sbert, # q 독일어
        #         attention_mask=q_sbert_mask, # q
        #         encoder_hidden_states=env_feat, # k, v 출발어(영어)
        #         use_cache=False,
        #         return_dict=True
        #     ).last_hidden_state
            
        #     env_feat = res_env + F.tanh(self.gamma) * out
        ########## query, env feat ##########
        
        ########## vid, env feat ##########
        if self.vid_env_arch == 'concat':
            env_feat = interp_env_feat(env_feat, B, T, mode='nearest')
            
            res = encoder_out_v
            encoder_out_v = torch.cat([encoder_out_v, env_feat], dim=2)
            encoder_out_v = self.vid_env_proj(encoder_out_v)
            encoder_out_v = torch.relu(encoder_out_v)
            # encoder_out_v = F.gelu(encoder_out_v)
            
            encoder_out_v = res + encoder_out_v
        ########## vid, env feat ##########
            
        return encoder_out_v
