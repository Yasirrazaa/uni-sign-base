#!/bin/bash

# Directory for outputs
output_dir="outputs/wlasl_enhanced"
mkdir -p $output_dir

# Choose checkpoint type
checkpoint_type=${1:-"rgb"}  # Options: rgb or pose

if [ "$checkpoint_type" = "rgb" ]; then
    echo "Using RGB-Pose checkpoint"
    ckpt_path="wlasl_rgb_pose_islr.pth"
    rgb_flag="--rgb_support"
else
    echo "Using Pose-only checkpoint"
    ckpt_path="wlasl_pose_only_islr.pth"
    rgb_flag=""
fi

# Set number of classes for WLASL dataset
num_classes=2000

# Set vocabulary path
vocab_path="data/WLASL/gloss_vocab.json"

# Enable enhanced model features
temporal_attn_flag="--use_temporal_attention"
fpn_flag="--use_fpn"
focal_loss_flag="--use_focal_loss"
mixup_flag="--use_mixup"
future_mask_flag="--use_future_mask"

# Training command with enhanced model and hyperparameters
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --epochs 30 \
    --opt AdamW \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --warmup-epochs 3 \
    --dropout 0.3 \
    --label-smoothing 0.1 \
    --clip-grad 1.0 \
    --output_dir $output_dir \
    --finetune $ckpt_path \
    --dataset WLASL \
    --task ISLR \
    --max_length 64 \
    --num_classes $num_classes \
    --vocab_path $vocab_path \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --mixup_alpha 0.2 \
    $rgb_flag \
    $temporal_attn_flag \
    $fpn_flag \
    $focal_loss_flag \
    $mixup_flag \
    $future_mask_flag

# Usage:
# For RGB-pose model: ./train_wlasl_enhanced.sh rgb
# For Pose-only model: ./train_wlasl_enhanced.sh pose
