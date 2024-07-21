import torch
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM, AutoTokenizer
from model.ours.nlq_head import NLQHead
import torch.nn as nn
from pathlib import Path
import json
from tqdm import tqdm
import random

class GroundVQA(nn.Module):
    def __init__(self, lm_path='google/flan-t5-base', input_dim=2304, freeze_word=False, max_v_len=256):
        super().__init__()

        if not isinstance(input_dim, int):
            input_dim = input_dim.v_dim

        self.lm: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(lm_path, local_files_only=True)

        lm_dim = self.lm.get_input_embeddings().embedding_dim
        self.lm_proj = nn.Linear(input_dim, lm_dim)
        self.v_emb = nn.Parameter(torch.randn((1, 1, lm_dim)))
        if freeze_word:
            for name, param in self.lm.named_parameters():
                if 'shared' in name:
                    param.requires_grad = False

        self.nlq_head = NLQHead(in_dim=lm_dim, max_v_len=max_v_len)
        
class NLQDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer, feature_type='global'):
        p_nlq_json = Path(f"/data/soyeonhong/GroundVQA/data/unified/annotations.NLQ_{split}.json")
        except_list = [
            '70a350cd-4f32-40ed-80dd-23a48e7c4e46',
            '04199001-307a-40fd-b20c-4b4128546b89',
            "486ab34d-48ac-4a8a-8895-62399abbb25e"] # nlq_train
        self.nlq_list = []
        for annotation in json.load(p_nlq_json.open()):
            video_id = annotation['video_id']
            if video_id in except_list:
                continue
            self.nlq_list.append(annotation)
        self.p_llava_caps_dir = Path(f"/data/soyeonhong/GroundVQA/llava-v1.6-34b/{feature_type}")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.nlq_list)

    def __getitem__(self, idx):
        annotation = self.nlq_list[idx]
        video_id = annotation['video_id']
        p_llava_cap = self.p_llava_caps_dir / f'{video_id}.json'
        llava_cap = json.load(p_llava_cap.open())['answers']

        caption_list = []
        frame_idx_list = []
        for frame_idx, _, caption in llava_cap:
            frame_idx_list.append(frame_idx)
            sentences = caption.split('. ')
            sentence_idx = random.randint(0, len(sentences) - 1)
            caption_list.append(sentences[sentence_idx])

        tokens = self.tokenizer(
            caption_list, return_tensors='pt', padding=True, truncation=False)

        return {
            'video_id': annotation['video_id'],
            'sample_id': annotation['sample_id'],
            'tokens': tokens,
            'frame_idx_list': frame_idx_list,
        }
        
def main():
    state_dict = torch.load("/data/soyeonhong/GroundVQA/checkpoints/GroundVQA_B-NLQ-VLG-val_R1_03=15.5.ckpt", map_location='cpu')['state_dict']
    
    model = GroundVQA().cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    lm = model.lm
    
    p_out_dir = Path(f"/data/soyeonhong/GroundVQA/all-mpnet-base-v2-tuned/flan_t5_one_sentence/captions")
    p_out_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val']:
    
        ds = NLQDataset(split, tokenizer)
        
        for annotation in tqdm(ds):
            video_id = annotation['video_id']
            tokens = annotation['tokens'].to('cuda')
            frame_idx_list = annotation['frame_idx_list']
            p_out = p_out_dir / f"{video_id}.pt"
            if p_out.exists():
                continue
            
            with torch.no_grad():
                cap_feat = lm.encoder.embed_tokens(tokens['input_ids'])
                
                out = lm.encoder(
                    inputs_embeds=cap_feat,
                    attention_mask=tokens['attention_mask'],
                ).last_hidden_state
                
                out = out.mean(dim=1).cpu()
                
            output_list = [(frame_idx, emb) for frame_idx, emb in zip(frame_idx_list, out)]
            
            torch.save(output_list, p_out)

if __name__ == '__main__':
    main()
    
