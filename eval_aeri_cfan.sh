#!/bin/bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data1/Datasets/ReID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_FILE="${CONFIG_FILE:-configs/aeri_cfan_baseline.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data1/weights/CFAN_weights/best0.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/AERI-PEDES/cfan_eval}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python test.py \
  --config_file "${CONFIG_FILE}" \
  --root_dir "${DATA_ROOT}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_finetune_model
