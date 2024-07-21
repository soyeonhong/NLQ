#!/bin/bash

#SBATCH --job-name gvqa_training_vid_concat_egovlp
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --partition batch_grad
#SBATCH -w  ariel-v8
#SBATCH -t 3-0
#SBATCH -e slurm/logs/slurm-%A_%x.err
#SBATCH -o slurm/logs/slurm-%A_%x.out

date +%Y-%m-%d/%H:%M:%S

if [[ $SLURM_JOB_PARTITION == *"batch"* ]]; then
    enable=False
else 
    enable=True
fi

##########Batch size and learning rate calculation##########
NUMBER_Of_GPUS=8
export base_lr=0.0001
export base_bsz=$(( $NUMBER_Of_GPUS * 16 ))
export bsz_per_gpu=6
export bsz=$(( $NUMBER_Of_GPUS * $bsz_per_gpu ))
export lr=$(python -c "print(f'{""$base_lr / $base_bsz * $bsz"":.5e}')")
echo "Base LR: $base_lr, Base BSZ: $base_bsz, LR: $lr, BSZ: $bsz"
##########Batch size and learning rate calculation##########

#  ['103204','102720',  '103329', '103250', '103327', /'103248', '103430', '103429', '104353', '103410', '104454']
######################Job_ID######################
job_id=103204
echo "Job ID: $job_id"
######################Job_ID######################

##########Env feature##########
# env_feature=/data/gunsbrother/prjs/ltvu/llms/GroundVQA/data/features/cheat_envs/01_cheat_env_bimodal_plus1 # have to change env_feature_type=cheating
# env_feature=/data/soyeonhong/GroundVQA/env_feature/$job_id/captions
# sbert_q_feat_path=/data/soyeonhong/GroundVQA/env_feature/$job_id/queries
# env_feature=/data/soyeonhong/GroundVQA/env_feature_pretrained/all-mpnet-base-v2/captions
# sbert_q_feat_path=/data/soyeonhong/GroundVQA/env_feature_pretrained/all-mpnet-base-v2/queries
env_feature=/data/soyeonhong/GroundVQA/env_feature_pretrained/distilbert-base-uncased/captions
sbert_q_feat_path=/data/soyeonhong/GroundVQA/env_feature_pretrained/distilbert-base-uncased/queries
echo "Env Feature: $env_feature"
echo "Query Feature: $sbert_q_feat_path"
##########Env feature##########

epoch=100

##########Training##########
python run.py \
    "model=groundvqa_b" \
    "dataset.nlq_train_splits=[NLQ_train]" \
    "dataset.test_splits=[NLQ_val]" \
    "dataset.batch_size=${bsz_per_gpu}" \
    "trainer.gpus=${NUMBER_Of_GPUS}" \
    "trainer.enable_progress_bar=$enable" \
    "optim.optimizer.lr=${lr}"  \
    "trainer.max_epochs=$epoch" \
    "dataset.env_feature.env_feature_path=$env_feature" \
    "dataset.env_feature.sbert_q_feat_path=$sbert_q_feat_path" \
    "model.vid_env_arch=concat" \
    "model.query_env_arch=None" \
    "model.vid_sum_arch=None"
    # "dataset.env_feature.env_feature_type=cheating"
##########Training##########

##########Inference##########
# python run.py \
#     model=groundvqa_b \
#     'dataset.nlq_train_splits=[NLQ_train]' \
#     'dataset.test_splits=[NLQ_val]' \
#     dataset.batch_size=32 \
#     +trainer.test_only=True \
#     "dataset.env_feature.env_feature_path=$env_feature" \
#     "dataset.env_feature.sbert_q_feat_path=$sbert_q_feat_path" \
#     '+trainer.checkpoint_path="/data/gunsbrother/prjs/ltvu/everything/NLQ-soyeon/lightning_logs/version_109924/checkpoints/step=6623-val_R1_03=13.467.ckpt"' \
#     "model.vid_env_arch=concat" \
#     "model.query_env_arch=ca" \
#     "model.vid_sum_arch=None"
##########Inference##########
