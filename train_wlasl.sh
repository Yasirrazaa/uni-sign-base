#!/bin/bash

# Directory for outputs
output_dir="outputs/wlasl_finetuning"
mkdir -p $output_dir

# Choose checkpoint type
checkpoint_type=${1:-"rgb"}  # Options: rgb or pose
use_future_mask=${2:-"false"}  # Options: true or false
use_classifier=${3:-"false"}  # Options: true or false

if [ "$checkpoint_type" = "rgb" ]; then
    echo "Using RGB-Pose checkpoint"
    ckpt_path="wlasl_rgb_pose_islr.pth"
    rgb_flag="--rgb_support"
else
    echo "Using Pose-only checkpoint"
    ckpt_path="wlasl_pose_only_islr.pth"
    rgb_flag=""
fi



# Classification head flag
if [ "$use_classifier" = "true" ]; then
    echo "Using classification head with frozen backbone"
    classifier_flags="--use_classifier_head --num_classes 2000 --freeze_backbone"
    # Higher learning rate when training only classifier head
    lr="1e-3"
else
    classifier_flags=""
    # Default learning rate for full model training
    lr="5e-5"
fi

# Training command - using fine_tuning.py instead of finetune_wlasl.py
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
    --batch-size 64 \
    --gradient-accumulation-steps 1 \
    --epochs 20 \
    --opt AdamW \
    --lr $lr \
    --warmup-epochs 2 \
    --output_dir $output_dir \
    --finetune $ckpt_path \
    --dataset WLASL \
    --task ISLR \
    $rgb_flag \
    $classifier_flags

# Usage:
# For RGB-pose model without future masking: ./train_wlasl.sh rgb false false
# For RGB-pose model with future masking: ./train_wlasl.sh rgb true false
# For Pose-only model without future masking: ./train_wlasl.sh pose false false
# For Pose-only model with future masking: ./train_wlasl.sh pose true false
#
# To use classification head, set the last parameter to true:
# For RGB-pose model with classification head: ./train_wlasl.sh rgb false true
