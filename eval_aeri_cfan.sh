#!/bin/bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/wuyong/datasets}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_FILE="${CONFIG_FILE:-configs/aeri_cfan_baseline.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/wuyong/data/weights/CFAN_weights/best0.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/AERI-PEDES/cfan_eval}"
USE_SWANLAB="${USE_SWANLAB:-0}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-CFAN}"
SWANLAB_EXPERIMENT="${SWANLAB_EXPERIMENT:-aeri_cfan_author_eval}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

args=(
  --config_file "${CONFIG_FILE}"
  --root_dir "${DATA_ROOT}"
  --checkpoint "${CHECKPOINT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --use_finetune_model
)

if [ "${USE_SWANLAB}" = "1" ]; then
  args+=(
    --use_swanlab
    --swanlab_project "${SWANLAB_PROJECT}"
    --swanlab_experiment "${SWANLAB_EXPERIMENT}"
    --swanlab_mode "${SWANLAB_MODE}"
  )
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python test.py "${args[@]}"
