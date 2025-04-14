#!/bin/bash
# Path to model checkpoint
ckpt_path=${1:-"outputs/wlasl_enhanced/best_checkpoint.pth"}

# Choose checkpoint type
checkpoint_type=${2:-"rgb"}  # Options: rgb or pose

# Set number of classes for WLASL dataset
num_classes=2000

# Set vocabulary path
vocab_path="data/WLASL/gloss_vocab.json"

# Set up output directory
if [ "$checkpoint_type" = "rgb" ]; then
    echo "Evaluating RGB-Pose enhanced model"
    rgb_flag="--rgb_support"
    output_dir="out/wlasl_enhanced_rgb_pose_eval"
else
    echo "Evaluating Pose-only enhanced model"
    rgb_flag=""
    output_dir="out/wlasl_enhanced_pose_only_eval"
fi

# Enable enhanced model features
temporal_attn_flag="--use_temporal_attention"
fpn_flag="--use_fpn"
future_mask_flag="--use_future_mask"

# Single GPU inference for enhanced model
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
   --batch-size 16 \
   --gradient-accumulation-steps 1 \
   --epochs 30 \
   --opt AdamW \
   --lr 1e-4 \
   --weight-decay 0.05 \
   --dropout 0.3 \
   --output_dir $output_dir \
   --finetune $ckpt_path \
   --dataset WLASL \
   --task ISLR \
   --max_length 64 \
   --num_classes $num_classes \
   --vocab_path $vocab_path \
   --eval \
   $rgb_flag \
   $temporal_attn_flag \
   $fpn_flag \
   $future_mask_flag

# Usage:
# Evaluate RGB-Pose model: ./script/eval_wlasl_enhanced.sh path/to/checkpoint.pth rgb
# Evaluate Pose-only model: ./script/eval_wlasl_enhanced.sh path/to/checkpoint.pth pose
