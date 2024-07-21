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
        self.nlq_list = []
        for annotation in json.load(p_nlq_json.open()):
            self.nlq_list.append(annotation)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.nlq_list)

    def __getitem__(self, idx):
        annotation = self.nlq_list[idx]
        question = annotation['question']

        tokens = self.tokenizer(
            question, return_tensors='pt', padding=True, truncation=False)

        return {
            'sample_id': annotation['sample_id'],
            'tokens': tokens,
        }
        
def main():
    
    state_dict = torch.load('/data/gunsbrother/prjs/ltvu/everything/sbert_finetune/data/checkpoints/egovlp-config-removed.pth', map_location='cpu')
    new_state = {}
    for k, v in state_dict['state_dict'].items():
        if k.startswith('module.text_model.'):
            new_state[k.replace('module.text_model.', '')] = v
            
    model = AutoModel.from_pretrained('distilbert-base-uncased', local_files_only=False, state_dict=new_state).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    p_out_dir = Path(f"/data/soyeonhong/GroundVQA/env_feature_pretrained/distilbert-base-uncased/queries")
    p_out_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val']:
    
        ds = NLQDataset(split, tokenizer)
        
        for annotation in tqdm(ds):
            sample_id = annotation['sample_id']
            tokens = annotation['tokens'].to('cuda')
            p_out = p_out_dir / f"{sample_id}.pt"

            if p_out.exists():
                continue
            
            with torch.no_grad():
                tokens = {k: v.cuda() for k, v in tokens.items()}
                embeddings = model(**tokens).last_hidden_state[:, 0]
            
            torch.save(embeddings.cpu(), p_out)

if __name__ == '__main__':
    main()       