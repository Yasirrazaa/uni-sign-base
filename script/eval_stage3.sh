#!/bin/bash

ckpt_path=wlasl_rgb_pose_islr.pth

# Default values
N_FOLDS=5
WANDB_PROJECT="uni-sign-eval"
WANDB_RUN_NAME="eval-$(date +%Y%m%d-%H%M%S)"

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --n_folds)
      N_FOLDS="$2"
      shift 2
      ;;
    --wandb_project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb_run_name)
      WANDB_RUN_NAME="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

# single gpu inference
# RGB-pose setting
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
   --batch-size 8 \
   --gradient-accumulation-steps 1 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --output_dir out/test \
   --finetune $ckpt_path \
   --dataset WLASL \
   --task ISLR \
   --eval \
   --rgb_support \
   --n_folds $N_FOLDS \
   --wandb_project "$WANDB_PROJECT" \
   --wandb_run_name "$WANDB_RUN_NAME"

# # pose-only setting
#deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
#   --batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --epochs 20 \
#   --opt AdamW \
#   --lr 3e-4 \
#   --output_dir out/test \
#   --finetune $ckpt_path \
#   --dataset CSL_Daily \
#   --task SLT \
#   --eval \


