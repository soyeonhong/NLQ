import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from pathlib import Path
import json
from tqdm import tqdm
import random

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
    
    state_dict = torch.load('/data/gunsbrother/prjs/ltvu/everything/sbert_finetune/data/checkpoints/egovlp-config-removed.pth', map_location='cpu')
    new_state = {}
    for k, v in state_dict['state_dict'].items():
        if k.startswith('module.text_model.'):
            new_state[k.replace('module.text_model.', '')] = v
            
    model = AutoModel.from_pretrained('distilbert-base-uncased', local_files_only=False, state_dict=new_state).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    p_out_dir = Path(f"/data/soyeonhong/GroundVQA/env_feature_pretrained/distilbert-base-uncased/captions")
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
                tokens = {k: v.cuda() for k, v in tokens.items()}
                embeddings = model(**tokens).last_hidden_state[:, 0].cpu()
                
            output_list = [(frame_idx, emb) for frame_idx, emb in zip(frame_idx_list, embeddings)]
            
            torch.save(output_list, p_out)

if __name__ == '__main__':
    main()       