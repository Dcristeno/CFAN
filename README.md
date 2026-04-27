# AERI Baseline

This branch is a minimal AERI-PEDES baseline. It removes the previous experimental modules and keeps only the plain aerial-ground-text training framework.

## Baseline Objective

`LOSS_NAMES=base` is the only supported training objective:

```text
base_loss =
  SDM(aerial, text)
+ SDM(ground, text)
+ SDM(aerial, ground)
```

No CDA, FTA, bridge, prototype, track memory, MoE, joint loss, k=2 sampling, heuristic sampling, or tile mixing is included.

## Train

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
USE_SWANLAB=1 \
RUN_NAME='aeri_base_fullsample' \
SWANLAB_EXPERIMENT='aeri_base_fullsample' \
SEED=1 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

On the default server setup, this is equivalent to:

```bash
bash finetune.sh
```

## Test

```bash
python test.py \
  --config_file logs/AERI-PEDES/<run_dir>/configs.yaml \
  --checkpoint logs/AERI-PEDES/<run_dir>/best0.pth \
  --root_dir /home/wuyong/datasets
```
