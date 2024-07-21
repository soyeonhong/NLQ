# NLQv2 val
CUDA_VISIBLE_DEVICES=0 python run.py \
    model=groundvqa_b \
    'dataset.qa_train_splits=[QaEgo4D_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=32 \
    +trainer.test_only=True \
    '+trainer.checkpoint_path="/data/soyeonhong/GroundVQA/lightning_logs/version_109431/checkpoints/step=6143-val_R1_03=16.337.ckpt"' \
    trainer.load_nlq_head=True