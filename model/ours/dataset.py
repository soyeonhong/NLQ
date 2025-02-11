# Joint dataset of CloseQA, OpenQA, and NLQ

import os
import math
import json
import random
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
import re

def interp_t(tensor, T_target, mode='nearest'):
    # tensor: [T_source, D]
    D, dtype = tensor.shape[-1], tensor.dtype
    return F.interpolate(tensor[None, None].float(), size=(T_target, D), mode=mode).squeeze([0, 1]).to(dtype=dtype)

def replace_func(text):
    # Define the pattern to match "." and "?"
    pattern = r'[.?]|Text:\w+|Query|Text:'
    # Replace occurrences of the pattern with an empty string
    result = re.sub(pattern, '', text)
    return result

class BaseDataset(Dataset):
    def __init__(self, data_dir, split, feature_type, max_v_len):
        super().__init__()
        self.split = split
        self.video_features = h5py.File(os.path.join(data_dir, feature_type + '.hdf5'), 'r')
        self.annotations = json.loads(Path(os.path.join(data_dir, f'annotations.{split}.json')).read_text())
        self.max_v_len = max_v_len
        print(f'{split} set: {len(self.annotations)}')
    
    def __len__(self):
        return len(self.annotations)
    
    def _get_video_feature(self, video_id):
        video_feature = torch.from_numpy(self.video_features[video_id][:])
        v_len = video_feature.shape[0]
        sample_ratio = 1.0
        if v_len > self.max_v_len:
            sample_idx = torch.linspace(0, v_len-1, self.max_v_len).long()
            video_feature = video_feature[sample_idx]
            sample_ratio = self.max_v_len / v_len
            v_len = self.max_v_len
        return video_feature, v_len, sample_ratio


class NLQDataset(BaseDataset):
    def __init__(self, data_dir, split, feature_type, max_v_len):
        super().__init__(data_dir, split, feature_type, max_v_len)

    def __getitem__(self, index):
        video_id = self.annotations[index]['video_id']
        query_id = self.annotations[index].get('sample_id')
        question = self.annotations[index]['question']

        video_feature, v_len, sample_ratio = self._get_video_feature(video_id)

        if 'clip_start_sec' in self.annotations[index]:
            start_time = self.annotations[index].get('clip_start_sec')
            end_time = self.annotations[index].get('clip_end_sec')
        else:
            start_time = self.annotations[index].get('moment_start_frame') / 30
            end_time = self.annotations[index].get('moment_end_frame') / 30

        query_type = self.annotations[index].get('query_type')
        if query_type == 'narration':
            duration = end_time - start_time
            center = (end_time + start_time) / 2
            scale_ratio = random.randint(1, 10)
            shift_number = random.uniform(-1, 1) * (scale_ratio - 1) * duration / 2
            new_center = center - shift_number
            start_time = new_center - scale_ratio * duration / 2
            end_time = new_center + scale_ratio * duration / 2

        segments = torch.tensor([[start_time, end_time]]) * 30 / 16.043 * sample_ratio
        labels = torch.zeros(len(segments), dtype=torch.int64)
        one_hot_labels = F.one_hot(labels, 1)  # (1, 1)

        return {
            'video_id': video_id,
            'question': f"question: {question} video: ",
            'answer': 'None',
            'v_feat': video_feature,
            'v_len': v_len,
            'segments': segments,
            'one_hot_labels': one_hot_labels,
            'query_id': query_id,
            'sample_ratio': sample_ratio,
            'task': 'NLQ'
        }
        
class NLQDatasetOnEnvfeature(NLQDataset):
    def __init__(self, data_dir, split, feature_type, max_v_len,
                 env_feature_path, sbert_q_feat_path, env_feature_type, 
                 subset, object, one_set):
        super().__init__(data_dir, split, feature_type, max_v_len)
        
        self.env_feature_path = env_feature_path
        self.one_set = one_set
        self.env_feature_type = env_feature_type
        self.sbert_q_feat_path = sbert_q_feat_path
        self.subset = subset
        self.object = object
        
        self.p_env_dir = Path(self.env_feature_path)
        
        if self.env_feature_type == 'llava_caption':
            required_clip_uids = set(a['video_id'] for a in self.annotations)
        elif self.env_feature_type == 'cheating':
            required_clip_uids = set(a['sample_id'] for a in self.annotations)

        valid_clip_uids = set(video_id.stem for video_id in list(self.p_env_dir.glob('**/*.pt')))
        
        diff = required_clip_uids - valid_clip_uids
        print(f'Clips not existing in LLaVA: {diff} ({len(diff)})')
        
        if self.env_feature_type == 'llava_caption':
            self.annotations = [a for a in self.annotations if a['video_id'] in valid_clip_uids]
        elif self.env_feature_type == 'cheating':
            self.annotations = [a for a in self.annotations if a['sample_id'] in valid_clip_uids]
            
        if self.one_set:
            self.annotations = self.annotations[:32]
        
    def __getitem__(self, index):
        output = super().__getitem__(index)
        
        video_id = output['video_id']
        query_id = output['query_id']
        v_len = output['v_len']
        
        
        if self.env_feature_type == 'llava_caption':
            env_feature = self.get_env_feature_llava(video_id, v_len)
        else:
            env_feature = self.get_env_feature_cheating(query_id, v_len)
        
        output['env_feat'] = env_feature
        output['q_sbert'] = torch.load(f"{self.sbert_q_feat_path}/{query_id}.pt")
        
        return output
        
    def get_env_feature_cheating(self, query_id, v_len):
        p_env_feature = self.p_env_dir / f'{query_id}.pt'
        env_feature = torch.load(p_env_feature, map_location='cpu')
        tensors = []
        for feature in env_feature:
            tensors.append(feature[-1])
        tensors = torch.stack(tensors) # [T_env_source, 768]
        
        tensors = interp_t(tensors, v_len)
        
        return tensors
    
    def get_env_feature_llava(self, query_id, v_len):
        p_env_feature = self.p_env_dir / f'{query_id}.pt'
        env_feature = torch.load(p_env_feature, map_location='cpu')
        tensors = []
        for feature in env_feature:
            tensors.append(feature[-1])
        tensors = torch.stack(tensors) # [T_env_source, 7168]
        
        return tensors

class QADataset(BaseDataset):
    def __init__(self, data_dir, split, feature_type, max_v_len, qa_type, CloseQA_weight=50):
        super().__init__(data_dir, split, feature_type, max_v_len)
        self.qa_type = qa_type  # CloseQA, OpenQA, Mixed
        self.choice_indices = ['A', 'B', 'C', 'D']
        self.CloseQA_weight = CloseQA_weight
        self.openqa_weight = 100 - CloseQA_weight

    def __getitem__(self, index):
        video_id = self.annotations[index]['video_id']
        query_id = self.annotations[index].get('sample_id')
        question = self.annotations[index]['question']
        answer = self.annotations[index]['answer'].strip()

        qa_type = self.qa_type
        if qa_type == 'Mixed':  # randomly choose a qa type
            qa_type = random.choices(['CloseQA', 'OpenQA'], weights=[self.CloseQA_weight, self.openqa_weight], k=1)[0]
        if qa_type == 'OpenQA':
            question_str = f"question: {question} video: "
            answer_str = answer
        elif qa_type == 'CloseQA':
            wrong_answers = self.annotations[index]['wrong_answers']
            # shuffle choices
            choices = [answer] + wrong_answers
            random.shuffle(choices)
            answer_index = choices.index(answer)
            choices = [f'({self.choice_indices[idx]}) {choices[idx]}' for idx in range(len(choices))]  # ["(A) xx", "(B) xx", "(C) xx", "(D) xx"]
            choices_str = ' '.join(choices)  # (A) xx (B) xx (C) xx (D) xx
            question_str = f"question: {question} choices: {choices_str}. video: "
            answer_str = choices[answer_index]  # (A/B/C/D) xx
        else:
            raise NotImplementedError
        
        video_feature, v_len, sample_ratio = self._get_video_feature(video_id)

        start_frame = self.annotations[index].get('moment_start_frame')
        end_frame = self.annotations[index].get('moment_end_frame')
        start_time = start_frame / 30
        end_time = end_frame / 30

        if 'video_start_sec' not in self.annotations[index]:  # LLM generated QA
            duration = end_time - start_time
            center = (end_time + start_time) / 2
            scale_ratio = random.randint(1, 10)
            shift_number = random.uniform(-1, 1) * (scale_ratio - 1) * duration / 2
            new_center = center - shift_number
            start_time = new_center - scale_ratio * duration / 2
            end_time = new_center + scale_ratio * duration / 2

        segments = torch.tensor([[start_time, end_time]]) * 30 / 16.043 * sample_ratio
        labels = torch.zeros(len(segments), dtype=torch.int64)
        one_hot_labels = F.one_hot(labels, 1)  # (1, 1)

        return {
            'video_id': video_id,
            'question': question_str,
            'answer': answer_str,
            'v_feat': video_feature,
            'v_len': v_len,
            'segments': segments,
            'one_hot_labels': one_hot_labels,
            'query_id': query_id,
            'sample_ratio': sample_ratio,
            'task': qa_type
        }


class JointDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset], tokenizer_path) -> None:
        super().__init__(datasets)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # BUG: Set this per convenience for GPT-2

    def collate_fn(self, batch):
        question = [b['question'] for b in batch]
        question_tok = self.tokenizer(question, padding=True, return_tensors='pt', add_special_tokens=False)
        
        answer = [b['answer'] for b in batch]
        labels = self.tokenizer(answer, padding=True, return_tensors='pt').input_ids
        # NOTE: NLQ data does not have an answer
        for idx, a in enumerate(answer):
            if a == 'None':
                labels[idx] = torch.ones_like(labels[idx]) * -100

        video_feature = [b['v_feat'] for b in batch]
        video_feature_padded = pad_sequence(video_feature, batch_first=True)
        video_mask = pad_sequence([torch.ones(len(v)) for v in video_feature], batch_first=True).bool()

        result = {
            'video_id': [b['video_id'] for b in batch],
            'q_text': question,
            'q_token': question_tok.input_ids,
            'q_mask': question_tok.attention_mask.bool(),
            'v_feat': video_feature_padded,
            'v_mask': video_mask,
            'v_len': np.asarray([b['v_len'] for b in batch], dtype=np.long),
            'gt_segments': torch.stack([b['segments'] for b in batch]),
            'gt_labels': torch.stack([b['one_hot_labels'] for b in batch]),
            'query_id': [b['query_id'] for b in batch],
            'sample_ratio': [b['sample_ratio'] for b in batch],
            'a_text': answer,
            'labels': labels,
            'task': [b['task'] for b in batch]
        }

        if 'env_feat' in batch[0]:
            env_feature = [b['env_feat'] for b in batch]
        
            q_sbert = [b['q_sbert'] for b in batch]
            q_sbert_pad = pad_sequence(q_sbert, batch_first=True)
            q_sbert_mask = pad_sequence([torch.ones(len(v)) for v in q_sbert], batch_first=True).bool()
        
            result['q_sbert'] = q_sbert_pad
            result['q_sbert_mask'] = q_sbert_mask
            result['env_feat'] = pad_sequence(env_feature, batch_first=True)
            result['env_mask'] = pad_sequence([torch.ones(len(v)) for v in env_feature], batch_first=True).bool()
            
        return result


class JointDataModule(pl.LightningDataModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def setup(self, stage=None):
        ds_kws = {}
        ds_kws |= self.config.get('env_feature', {})
        
        CloseQA_weight = self.config.get('closeqa_weight', 50)
        print(f'CloseQA percentage: {CloseQA_weight}%')
        self.train_dataset = JointDataset([
                QADataset('data/unified', train_split, self.config.feature_type, self.config.max_v_len, 'Mixed', CloseQA_weight)
                for train_split in self.config.qa_train_splits
            ] + [
                NLQDatasetOnEnvfeature('data/unified', train_split, self.config.feature_type, self.config.max_v_len, **ds_kws)
                for train_split in self.config.nlq_train_splits
            ],
            self.config.tokenizer_path
        )

        test_datasets = []
        for split in self.config.test_splits:
            if split == 'QaEgo4D_test':
                test_datasets.append(QADataset('data/unified', split, self.config.feature_type, self.config.max_v_len, 'OpenQA'))
            elif split == 'QaEgo4D_test_close':
                test_datasets.append(QADataset('data/unified', split, self.config.feature_type, self.config.max_v_len, 'CloseQA'))
            elif split in ['NLQ_val', 'NLQ_test_unannotated']:
                test_datasets.append(NLQDatasetOnEnvfeature('data/unified', split, self.config.feature_type, self.config.max_v_len, **ds_kws))
            else:
                print(split)
                raise NotImplementedError
        self.val_dataset = self.test_dataset = JointDataset(test_datasets, self.config.tokenizer_path)

        print(f'#total train: {len(self.train_dataset)}')
        print(f'#total val: {len(self.val_dataset)}')
        print(f'#total test: {len(self.test_dataset)}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )
