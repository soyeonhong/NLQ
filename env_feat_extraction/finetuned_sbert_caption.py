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

# /data/gunsbrother/prjs/ltvu/everything/sbert_finetune/outputs/batch/2024-06-12/17-54-41/lit/104353/checkpoints/step=3834-nlq_R5@0.3=6.3928.ckpt
# /data/gunsbrother/prjs/ltvu/everything/sbert_finetune/outputs/batch/2024-06-08/18-40-03/lit/103410/checkpoints/step=4752-nlq_R5@0.3=0.0000.ckpt
@torch.no_grad()
def main():
    
    # for job_id in ['103204','102720',  '103329', '103250', '103327', '103248', '103430', '103429']:
    # for job_id in ['103248', '103430', '103429']:
    # for job_id in  ['103345', '103410', '103411']:
    # for job_id in ['103415']:
    # for job_id in ['104353', '103410']:
    for job_id in ['103204']:
        target_jobid = job_id
        print(f"Job ID: {target_jobid}")
        command = f"find /data/gunsbrother/prjs/ltvu/everything/sbert_finetune/outputs -name \"{target_jobid}\" -type d -exec bash -c 'find {{}} -name *.ckpt -type f' \\; -quit"

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        model_path = result.stdout.strip()
        
        if job_id == '104353':
            model_path = '/data/gunsbrother/prjs/ltvu/everything/sbert_finetune/outputs/batch/2024-06-12/17-54-41/lit/104353/checkpoints/step=3834-nlq_R5@0.3=6.3928.ckpt'
        elif job_id == '104454':
            model_path = '/data/gunsbrother/prjs/ltvu/everything/sbert_finetune/outputs/batch/2024-06-13/16-12-17/lit/104454/checkpoints/step=3051-nlq_R5@0.3=7.7109.ckpt'
        print(f"Model path: {model_path}")
        model = SentenceTransformer('all-mpnet-base-v2').cuda().eval()
        # model = SentenceTransformer('distilbert-base-uncased').cuda().eval()
        # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda().eval()
        model.load_state_dict(
            {k.replace('model.model.', '0.auto_model.'): v for k, v in torch.load(model_path)['state_dict'].items()},
            strict=False
        )
        print('Model loaded.')
        
        p_caps_dir = Path('/data/gunsbrother/prjs/ltvu/everything/sbert_finetune/data/captions/llava-v1.6-34b/global')
        p_out_dir = Path(f"/data/soyeonhong/GroundVQA/all-mpnet-base-v2-tuned_one_sentence/{target_jobid}/captions")
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
            # caps = [entry[2] for entry in cap_data]
            caps = []
            for entry in cap_data:
                sentences = entry[2].split('. ')
                entry_idx = random.randint(0, len(sentences) - 1)
                caps.append(sentences[entry_idx])
                
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
