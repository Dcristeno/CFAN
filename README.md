# CFAN Baseline Cleanup

This repo snapshot is a cleaned working copy of the author-provided CFAN baseline for `AERI-PEDES`.

## What was cleaned

- Removed hardcoded `CUDA_VISIBLE_DEVICES` from `finetune.py` and `test.py`.
- Added a safer evaluation entry that can load the author checkpoint directly.
- Normalized the legacy author loss naming `sdm+fa` to the finetune code's actual implementation names `cda+fta`.
- Added runnable shell scripts for finetuning and evaluation.

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

## Finetune CFAN on AERI-PEDES

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=pretrain/HAMbest0.pth \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

## Notes on the provided weights

The provided checkpoint contains finetune-only modules such as `query` and `mlp_logsigma2`, so evaluation should use `build_finetune_model`. The cleaned `test.py` now auto-detects this from the checkpoint.
