#!/bin/bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-Testing}"
DATA_ROOT="${DATA_ROOT:-/data0/wentao/data/textReID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python train.py \
  --name Pretrain \
  --img_aug \
  --batch_size 512 \
  --MLM \
  --dataset_name "${DATASET_NAME}" \
  --loss_names 'sdm' \
  --num_epoch 30 \
  --root_dir "${DATA_ROOT}" \
  --pretrain LuPerson_PEDES \
  --nam
