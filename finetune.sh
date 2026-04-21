#!/bin/bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-AERI-PEDES}"
DATA_ROOT="${DATA_ROOT:-/home/wuyong/datasets}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FINETUNE_INIT="${FINETUNE_INIT:-/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth}"
RUN_NAME="${RUN_NAME:-cfan_finetune}"
LOSS_NAMES="${LOSS_NAMES:-cda+fta}"
TRAIN_SAMPLES_PER_ID="${TRAIN_SAMPLES_PER_ID:-0}"
BRIDGE_LOSS_WEIGHT="${BRIDGE_LOSS_WEIGHT:-2.0}"
BRIDGE_GROUND_TEXT_WEIGHT="${BRIDGE_GROUND_TEXT_WEIGHT:-1.0}"
BRIDGE_GROUND_AERIAL_WEIGHT="${BRIDGE_GROUND_AERIAL_WEIGHT:-2.0}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-CFAN}"
SWANLAB_EXPERIMENT="${SWANLAB_EXPERIMENT:-cfan_aeri_finetune}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python finetune.py \
  --name "${RUN_NAME}" \
  --img_aug \
  --batch_size 64 \
  --MLM \
  --dataset_name "${DATASET_NAME}" \
  --loss_names "${LOSS_NAMES}" \
  --train_samples_per_id "${TRAIN_SAMPLES_PER_ID}" \
  --bridge_loss_weight "${BRIDGE_LOSS_WEIGHT}" \
  --bridge_ground_text_weight "${BRIDGE_GROUND_TEXT_WEIGHT}" \
  --bridge_ground_aerial_weight "${BRIDGE_GROUND_AERIAL_WEIGHT}" \
  --lr 5e-6 \
  --lr2 5e-5 \
  --num_epoch 60 \
  --root_dir "${DATA_ROOT}" \
  --use_swanlab \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment "${SWANLAB_EXPERIMENT}" \
  --swanlab_mode "${SWANLAB_MODE}" \
  --finetune "${FINETUNE_INIT}"
