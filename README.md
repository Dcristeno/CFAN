# CFAN Baseline Cleanup

This repo snapshot is a cleaned working copy of the author-provided CFAN baseline for `AERI-PEDES`.

## What was cleaned

- Removed hardcoded `CUDA_VISIBLE_DEVICES` from `finetune.py` and `test.py`.
- Added a safer evaluation entry that can load the author checkpoint directly.
- Normalized the legacy author loss naming `sdm+fa` to the finetune code's actual implementation names `cda+fta`.
- Added runnable shell scripts for finetuning and evaluation.
- Made SwanLab logging opt-in in both evaluation and finetuning scripts, so training does not stall on network reconnects by default.

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
To enable SwanLab logging for evaluation, add `USE_SWANLAB=1`.

## Finetune CFAN on AERI-PEDES

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

To enable SwanLab logging for finetuning, add `USE_SWANLAB=1`.

To run the cleaner `CDA+FTA` baseline with AERI per-ID sparse sampling, for example `k=2`, run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda+fta' \
TRAIN_SAMPLES_PER_ID=2 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

To replace random `k=2` sampling with a simple heuristic selector that keeps the sharpest aerial images per ID, run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='sharpness_topk' \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

To use a safer heuristic that prefers medium-sharpness aerial images, penalizes redundancy, and keeps more representative views per ID, run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='mid_sharpness_diverse' \
TRAIN_SAMPLE_MID_RATIO=0.6 \
TRAIN_SAMPLE_REPRESENTATIVE_WEIGHT=1.0 \
TRAIN_SAMPLE_DIVERSITY_WEIGHT=0.5 \
TRAIN_SAMPLE_MID_WEIGHT=0.5 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

To keep the strong random `k=2` baseline while discouraging near-duplicate aerial frames, run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='random_diverse' \
TRAIN_SAMPLE_DIVERSITY_WEIGHT=1.0 \
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
