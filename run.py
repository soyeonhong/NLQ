import re
import math
from argparse import ArgumentParser, Namespace

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, open_dict
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.plugins import DDPPlugin

from model.ours.dataset import JointDataModule
from model.ours.lightning_module import LightningModule


def dict_parser(s: str):
    return eval('{' + re.sub(r'(\w+)=(["\']?\w+["\']?)', r'"\1":\2', s) + '}')

def add_common_trainer_util_args(parser, default_monitor_variable='val_loss', default_monitor_mode='min'):
    if default_monitor_mode not in ['min', 'max']:
        raise ValueError(default_monitor_mode)
    parser.add_argument('--lr_find_kwargs', default=dict(min_lr=5e-6, max_lr=1e-2), type=dict_parser,
                        help='Arguments for LR find (--auto_lr_find). Default "min_lr=5e-6,max_lr=1e-2"')
    parser.add_argument('--random_seed', default=42, type=lambda s: None if s == 'None' else int(s),
                        help='Seed everything. Set to "None" to disable global seeding')
    parser.add_argument('--auto_resume', default=False, action='store_true',
                        help='Automatically resume last saved checkpoint, if available.')
    parser.add_argument('--test_only', default=False, action='store_true',
                        help='Skip fit and call only test. This implies automatically detecting newest checkpoint, '
                             'if --checkpoint_path is not given.')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Load this checkpoint to resume training or run testing. '
                             'Pass in the special value "best" to use the best checkpoint according to '
                             'args.monitor_variable and args.monitor_mode. '
                             'Using "best" only works with test_only mode.')
    parser.add_argument('--ignore_existing_checkpoints', default=False, action='store_true',
                        help='Proceed even with training a new model, even if previous checkpoints exists.')
    parser.add_argument('--monitor_variable', default=default_monitor_variable, type=str,
                        help='Variable to monitor for early stopping and for checkpoint selection. '
                             f'Default: {default_monitor_variable}')
    parser.add_argument('--monitor_mode', default=default_monitor_mode, type=str, choices=['min', 'max'],
                        help='Mode for monitoring the monitor_variable (for early stopping and checkpoint selection). '
                             f'Default: {default_monitor_mode}')
    parser.add_argument('--reset_early_stopping_criterion', default=False, action='store_true',
                        help='Reset the early stopping criterion when loading from checkpoint. '
                             'Prevents immediate exit after switching to more complex dataset in curriculum strategy')

def apply_argparse_defaults_to_hydra_config(config: DictConfig, parser: ArgumentParser, verbose=False):
    args = parser.parse_args([])  # Parser is not allowed to have required args, otherwise this will fail!
    defaults = vars(args)

    def _apply_defaults(dest: DictConfig, source: dict, indentation=''):
        for k, v in source.items():
            if k in dest and isinstance(v, dict):
                current_value = dest[k]
                if current_value is not None:
                    assert isinstance(current_value, DictConfig)
                    _apply_defaults(current_value, v, indentation + ' ')
            elif k not in dest:
                dest[k] = v
                if verbose:
                    print(indentation, 'set default value for', k)

    with open_dict(config):
        _apply_defaults(config, defaults)


def _adjust_ddp_config(trainer_cfg):
    trainer_cfg = dict(trainer_cfg)
    strategy = trainer_cfg.get('strategy', None)
    if trainer_cfg['gpus'] > 1 and strategy is None:
        strategy = 'ddp'  # Select ddp by default
    if strategy == 'ddp':
        trainer_cfg['strategy'] = DDPPlugin(
            find_unused_parameters=trainer_cfg['find_unused_parameters'], 
            gradient_as_bucket_view=True)
    return trainer_cfg


@hydra.main(config_path='config', config_name='base')
def train(config: DictConfig):
    fake_parser = ArgumentParser()
    add_common_trainer_util_args(fake_parser, default_monitor_variable='val_loss')
    apply_argparse_defaults_to_hydra_config(config.trainer, fake_parser)
    pl.seed_everything(config.trainer.random_seed, workers=True)
    trainer_cfg = Namespace(**_adjust_ddp_config(config.trainer))

    data = JointDataModule(config.dataset)
    data.setup()

    total_steps = trainer_cfg.max_epochs * math.floor(len(data.train_dataset) / trainer_cfg.gpus / config.dataset.batch_size)
    model = LightningModule(config, total_steps)
    if trainer_cfg.checkpoint_path:
        print(f'Load checkpoint: {trainer_cfg.checkpoint_path}')
        state_dict = torch.load(trainer_cfg.checkpoint_path, map_location='cpu')['state_dict']
        if not trainer_cfg.test_only:
            print("=" * 50)
            param_names = ['vid_t_proj', 'nlq_head', 'env_q_sbert_attn', 'gamma', 'env_emb',  'vid_sum_env_proj',  'vid_sum_emb', 'vid_sum_proj', 'vid_sum_attn', 'gamma2', 'vid_sum_env_attn', 'vid_sum_env_attn', 'alpha', 'vid_env_proj', 'vid_sum_proj', 'query_env_proj']

            for name, param in model.named_parameters():
                # Check if the current parameter name contains any of the specified substrings
                if any(substring in name for substring in param_names):
                    if 'shared' in name:
                        param.requires_grad = False
                    else:
                        print(name)
                        param.requires_grad = True
                else:
                    param.requires_grad = False
            
            print("=" * 50)
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f'Missing Keys: {missing_keys}')
        print(f'Unexpected Keys: {unexpected_keys}')
    else:
        print('Train from scratch')
    
    print(model)


    if trainer_cfg.test_only:  # evaluation
        trainer = pl.Trainer.from_argparse_args(
            trainer_cfg, 
            enable_checkpointing=False, 
            logger=False
        )
        if trainer_cfg.val:
            trainer.validate(
                model, data.val_dataloader(),
            )
        else:
            trainer.test(
                model, [data.val_dataloader(), data.train_dataloader()],
            )
    else:  # training
        model_checkpoint = []
        if 'QaEgo4D_test' in config.dataset.test_splits:
            model_checkpoint.append(
                ModelCheckpoint(
                    save_last=False, 
                    monitor='val_ROUGE', 
                    mode='max',
                    save_top_k=1, 
                    filename='{step}-{' + 'val_ROUGE' + ':.3f}')
            )
        if 'QaEgo4D_test_close' in config.dataset.test_splits:
            model_checkpoint.append(
                ModelCheckpoint(
                    save_last=False, 
                    monitor='val_close_acc', 
                    mode='max',
                    save_top_k=1, 
                    filename='{step}-{' + 'val_close_acc' + ':.3f}')
            )
        if 'NLQ_val' in config.dataset.test_splits:
            model_checkpoint.append(
                ModelCheckpoint(
                    save_last=False, 
                    monitor='val_R1_03', 
                    mode='max',
                    save_top_k=1, 
                    filename='{step}-{' + 'val_R1_03' + ':.3f}')
            )
        trainer = pl.Trainer.from_argparse_args(trainer_cfg, callbacks=[
            LearningRateMonitor(logging_interval='step'),
            # StochasticWeightAveraging(swa_lrs=1e-2),
            *model_checkpoint
        ])
        trainer.fit(
            model, data.train_dataloader(), [data.val_dataloader(), data.train_dataloader()], 
        )
        
        
        # trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        #     trainer_cfg,
        #     callbacks=[
        #         ModelSummary(max_depth=2),
        #         LearningRateMonitor(logging_interval='step'),
        #         # StochasticWeightAveraging(swa_lrs=1e-2),
        #         *model_checkpoint
        #     ],
        #     logger=TensorBoardLogger(
        #         save_dir=trainer_cfg.default_root_dir,
        #         version=os.environ.get("SLURM_JOB_ID"),
        #         name="lit",
        #         # sub_dir='tb',
        #         default_hp_metric=False
        #     )
        # )

    
if __name__ == '__main__':
    train()
