import os
import json
import argparse
from pathlib import Path
import multiprocessing as mp

import torch
import torch.utils.data
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import subprocess

class NLQDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        p_nlq_json = Path(f"/data/soyeonhong/GroundVQA/data/unified/annotations.NLQ_{split}.json")
        self.nlq_list = []
        for annotation in json.load(p_nlq_json.open()):
            self.nlq_list.append(annotation)

    def __len__(self):
        return len(self.nlq_list)

    def __getitem__(self, idx):
        annotation = self.nlq_list[idx]
        question = annotation['question']

        return {'sample_id': annotation['sample_id'],
                'question': question}

def main():
    cmd = "scontrol show jobid ${SLURM_JOB_ID} | grep -oP '(?<=BatchFlag=)([0-1])'"
    batch_flag = int(os.popen(cmd).read().strip())
    disable = batch_flag == 1
    if not disable:
        print("BatchFlag is 0. tqdm is enabled.")
    
    # for job_id in ['102720', '103204', '103329', '103250', '103327', '103248', '103430', '103429']:
    # for job_id in ['103248', '103430', '103429']:
    # for job_id in  ['103345', '103410', '103411']:
    # for job_id in ['104353', '103410']:
    for job_id in ['104454']:

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
        # model = SentenceTransformer('all-mpnet-base-v2').cuda().eval()
        # model = SentenceTransformer('distilbert-base-uncased').cuda().eval()
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda().eval()
        model.load_state_dict(
            {k.replace('model.model.', '0.auto_model.'): v for k, v in torch.load(model_path)['state_dict'].items()},
            strict=False
        )
        print('Model loaded.')

        p_out_dir = Path(f"/data/soyeonhong/GroundVQA/all-mpnet-base-v2-tuned/{target_jobid}/queries")
        p_out_dir.mkdir(parents=True, exist_ok=True)
            
        for split in ['train', 'val']:
            
            ds = NLQDataset(split=split)
            
            print(f"{split} start")    
            for annotation in tqdm(ds, disable=disable):
                sample_id = annotation['sample_id']
                question = annotation['question']
                
                p_out = p_out_dir / f"{sample_id}.pt"

                if p_out.exists():
                    continue
                    
                encoder_output = torch.tensor(model.encode([question]))
                # encoder_output = torch.mean(encoder_output, dim = 0) # [# of object, 768] -> [768]
                        
                encoder_output = encoder_output.cpu().detach() # [768] -> [1, 768]

                torch.save(encoder_output, p_out)



if __name__ == '__main__':
    main()
