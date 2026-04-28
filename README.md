# AERI Baseline

This branch is a minimal AERI-PEDES baseline. It removes the previous experimental modules and keeps only the plain aerial-ground-text training framework.

## Environment

Conda is recommended:

```bash
conda env create -f environment.yml
conda activate irra
```

If PyTorch needs to match a different CUDA version on a server, install the matching PyTorch build first, then install the remaining Python packages with:

```bash
pip install -r requirements.txt
```

The pinned configuration mirrors the verified server setup:

```text
Python 3.8.20
PyTorch 2.0.0 / CUDA 11.8
TorchVision 0.15.0
SwanLab 0.7.15
```

## Baseline Objective

`LOSS_NAMES=base` is the only supported training objective:

```text
base_loss =
  SDM(aerial, text)
+ SDM(ground, text)
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
