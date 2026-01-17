#!/bin/bash

# Activate conda environment
source /DATA/guantianrui/anaconda3/etc/profile.d/conda.sh
conda activate watermark_anything

# Verify Python environment
echo "Using Python: $(which python)"
echo "Python path: $(python --version)"

# Use python -m torch.distributed.run instead of torchrun to ensure correct Python environment
python -m torch.distributed.run --nproc_per_node=8 train.py \
    --local_rank 0 --debug_slurm \
    --output_dir output_finetune \
    --augmentation_config configs/all_augs.yaml \
    --extractor_model sam_base \
    --embedder_model vae_small \
    --img_size 256 \
    --batch_size 16 \
    --batch_size_eval 32 \
    --epochs 100 \
    --optimizer "AdamW,lr=1e-4" \
    --scheduler "CosineLRScheduler,lr_min=1e-6,t_initial=100,warmup_lr_init=1e-6,warmup_t=5" \
    --seed 42 \
    --perceptual_loss lpips \
    --lambda_i 1.0 \
    --lambda_d 0.5 \
    --lambda_w 1.0 \
    --lambda_w2 10.0 \
    --nbits 32 \
    --scaling_i 1.0 \
    --scaling_w 1.0 \
    --resume_from /home/guantianrui/watermark-anything/checkpoints/wam_mit.pth \
    --train_dir  /share_data/guantianrui/datasets/COCO/train2017 --train_annotation_file  /share_data/guantianrui/datasets/COCO/annotations/instances_train2017.json \
    --val_dir /share_data/guantianrui/datasets/COCO/val2017 --val_annotation_file /share_data/guantianrui/datasets/COCO/annotations/instances_val2017.json \
    --use_wandb True