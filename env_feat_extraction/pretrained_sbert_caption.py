import sys
import json
from pathlib import Path
from tqdm import tqdm
import argparse

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import subprocess
import random

@torch.no_grad()
def main():
    
    model_name = 'all-mpnet-base-v2'

    model = SentenceTransformer(model_name).cuda().eval()

    
    p_caps_dir = Path('/data/gunsbrother/prjs/ltvu/everything/sbert_finetune/data/captions/llava-v1.6-34b/global')
    p_out_dir = Path(f"/data/soyeonhong/GroundVQA/env_feature_pretrained/{model_name}/captions")
    p_out_dir.mkdir(parents=True, exist_ok=True)

    p_caps = list(p_caps_dir.glob('*.json'))
    print(f'Found {len(p_caps)} files.')

    print('Save dir: ', p_out_dir)
    print('\n')
    num_caps_cum = 0
    pbar = tqdm(p_caps, total=len(p_caps), dynamic_ncols=True)
    for p_cap in pbar:
        cap_data = json.load(p_cap.open())['answers']
        p_out = p_out_dir / p_cap.with_suffix('.pt').name
        if p_out.exists():
            data = torch.load(p_out)
            num_caps = len(data)
            if num_caps == len(cap_data):
                sys.stdout.write('\033[A\033[2K  \033[A')
                sys.stdout.write(f'{p_out.name} already exists. Skipping.')
                sys.stdout.write('\033[B')
                num_caps_cum += num_caps
                pbar.set_postfix(num_caps=num_caps, num_caps_cum=num_caps_cum)
                continue

        frame_idxs = [entry[0] for entry in cap_data]
        caps = [entry[2] for entry in cap_data]
        # caps = []
        # for entry in cap_data:
        #     sentences = entry[2].split('. ')
        #     entry_idx = random.randint(0, len(sentences) - 1)
        #     caps.append(sentences[entry_idx])
            
        embeddings = model.encode(caps, convert_to_tensor=True, convert_to_numpy=False).cpu()  # [T, D]
        output_list = [(frame_idx, emb) for frame_idx, emb in zip(frame_idxs, embeddings)]
        torch.save(output_list, p_out)
        num_caps = len(caps)
        num_caps_cum += num_caps
        pbar.set_postfix(num_caps=num_caps, num_caps_cum=num_caps_cum)
        # tqdm.write(f'Saved to {p_out}.')
    print(f'Finished. Total {num_caps_cum} captions. Saved to {p_out_dir}.')


if __name__ == '__main__':
    main()