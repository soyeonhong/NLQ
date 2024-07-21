import os
import json
import argparse
from pathlib import Path
import multiprocessing as mp

import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


class NLQDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer, feature_type):
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
        question = annotation['question']
        p_llava_cap = self.p_llava_caps_dir / f'{video_id}.json'
        llava_cap = json.load(p_llava_cap.open())['answers']

        caption_list = []
        query_list = []
        frame_idx_list = []
        for frame_idx, _, caption in llava_cap:
            frame_idx_list.append(frame_idx)
            caption_list.append(caption)
            query_list.append(question)

        tokens = self.tokenizer(
            query_list, caption_list, return_tensors='pt', padding=True, truncation=False)

        return {
            'video_id': annotation['video_id'],
            'sample_id': annotation['sample_id'],
            'question': annotation['question'],
            'tokens': tokens,
            'frame_idx_list': frame_idx_list,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--dir', type=str, default='cross_encoding_truncation_false')
    parser.add_argument('--feature_type', type=str, default='global')
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds = NLQDataset(split=args.split, tokenizer=tokenizer, feature_type=args.feature_type)

    p_out_root_dir = Path(args.dir)
    p_out_dir = p_out_root_dir / args.split
    p_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = "scontrol show jobid ${SLURM_JOB_ID} | grep -oP '(?<=BatchFlag=)([0-1])'"
    batch_flag = int(os.popen(cmd).read().strip())
    disable = batch_flag == 1
    if not disable:
        print("BatchFlag is 0. tqdm is enabled.")

    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().cuda()
    for annotation in tqdm(ds, disable=disable):
        sample_id = annotation['sample_id']
        tokens = annotation['tokens'].to('cuda')
        frame_idx_list = annotation['frame_idx_list']
        p_out = p_out_dir / f"{sample_id}.pt"
        if p_out.exists():
            continue

        encoder_output = model(**tokens, return_dict=True, output_hidden_states=True)
        pool_result = model.bert.pooler(encoder_output.hidden_states[-1]).cpu().detach()

        save_list = list(zip(frame_idx_list, pool_result))
        torch.save(save_list, p_out)


if __name__ == '__main__':
    main()
