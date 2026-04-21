# CFAN Baseline Cleanup

This repo snapshot is a cleaned working copy of the author-provided CFAN baseline for `AERI-PEDES`.

## What was cleaned

- Removed hardcoded `CUDA_VISIBLE_DEVICES` from `finetune.py` and `test.py`.
- Added a safer evaluation entry that can load the author checkpoint directly.
- Normalized the legacy author loss naming `sdm+fa` to the finetune code's actual implementation names `cda+fta`.
- Added runnable shell scripts for finetuning and evaluation.
- Enabled SwanLab logging in both evaluation and finetuning scripts by default.

## Expected dataset layout

Set `DATA_ROOT` to the parent directory that contains `AERI-PEDES`, for example:

```bash
/home/wuyong/datasets/AERI-PEDES
```

Inside `AERI-PEDES`, CFAN expects at least:

- `train_caption.json`
- `test_caption.json`
- image folders referenced by those json files

## Evaluate the provided author checkpoint

```bash
DATA_ROOT=/home/wuyong/datasets \
CHECKPOINT_PATH=/home/wuyong/data/weights/CFAN_weights/best0.pth \
CUDA_VISIBLE_DEVICES=0 \
bash eval_aeri_cfan.sh
```

The default config used by this script is [configs/aeri_cfan_baseline.yaml](configs/aeri_cfan_baseline.yaml).
It logs test metrics to SwanLab by default with experiment name `aeri_cfan_author_eval`.

## Finetune CFAN on AERI-PEDES

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

This script logs training and validation metrics to SwanLab by default with experiment name `cfan_aeri_finetune`.

To run the cleaner `CDA+FTA` baseline with AERI per-ID sparse sampling, for example `k=2`, run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda+fta' \
TRAIN_SAMPLES_PER_ID=2 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

To enable the stronger ground-to-aerial bridge loss for AERI experiments, run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda+fta+bridge' \
BRIDGE_LOSS_WEIGHT=2.0 \
BRIDGE_GROUND_TEXT_WEIGHT=1.0 \
BRIDGE_GROUND_AERIAL_WEIGHT=2.0 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

The bridge loss first aligns `text <-> ground`, then uses `ground -> aerial` supervision to pull aerial features toward the cleaner ground-view space.

## Notes on the provided weights

The provided checkpoint contains finetune-only modules such as `query` and `mlp_logsigma2`, so evaluation should use `build_finetune_model`. The cleaned `test.py` now auto-detects this from the checkpoint.
