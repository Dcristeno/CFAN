import argparse
import os
import os.path as op
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from datasets.aeripedes import AERIPEDES
from datasets.bases import ImageDataset
from datasets.build import build_transforms
from model.build_finetune import build_finetune_model


def parse_args():
    parser = argparse.ArgumentParser(description="Build fixed aerial trajectory prototypes for AERI-PEDES training IDs")
    parser.add_argument("--root_dir", type=str, required=True, help="Dataset root that contains the AERI-PEDES folder")
    parser.add_argument("--finetune", type=str, required=True, help="Checkpoint used as the fixed feature extractor teacher")
    parser.add_argument("--output", type=str, required=True, help="Path to save the prototype tensor file")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for offline feature extraction")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--pretrain_choice", type=str, default="ViT-B/16")
    parser.add_argument("--img_size", type=int, nargs=2, default=(384, 128), metavar=("H", "W"))
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--loss_names", type=str, default="cda", help="Dummy finetune loss setting used only to build the backbone")
    parser.add_argument("--cmt_depth", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=49408)
    return parser.parse_args()


def load_finetune_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a 'model' state dict.")

    param_dict = checkpoint["model"]
    cleaned_state_dict = {}
    for key, value in param_dict.items():
        cleaned_state_dict[key.replace("module.", "")] = value.detach().clone()
    model.load_state_dict(cleaned_state_dict, strict=False)


def main():
    args = parse_args()
    os.makedirs(op.dirname(op.abspath(args.output)), exist_ok=True)

    dataset = AERIPEDES(root=args.root_dir, verbose=True)
    image_pids = [sample[0] for sample in dataset.train]
    img_paths = [sample[1] for sample in dataset.train]

    if not image_pids:
        raise RuntimeError("No training aerial samples were found for prototype extraction.")

    transform = build_transforms(img_size=tuple(args.img_size), is_train=False)
    image_set = ImageDataset(image_pids, img_paths, transform=transform)
    image_loader = DataLoader(
        image_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model_args = SimpleNamespace(
        pretrain_choice=args.pretrain_choice,
        img_size=tuple(args.img_size),
        stride_size=args.stride_size,
        temperature=args.temperature,
        loss_names=args.loss_names,
        cmt_depth=args.cmt_depth,
        vocab_size=args.vocab_size,
    )
    model = build_finetune_model(model_args, num_classes=len(set(image_pids)))
    load_finetune_checkpoint(model, args.finetune)
    model = model.float().cuda().eval()

    max_pid = max(image_pids)
    num_pids = max_pid + 1
    feature_sums = None
    counts = torch.zeros(num_pids, dtype=torch.long)

    with torch.no_grad():
        for pids, images in image_loader:
            images = images.cuda(non_blocking=True)
            feats = model.encode_image(images).cpu()

            if feature_sums is None:
                feature_sums = torch.zeros(num_pids, feats.shape[1], dtype=torch.float32)

            for pid, feat in zip(pids.tolist(), feats):
                feature_sums[pid] += feat.float()
                counts[pid] += 1

    if feature_sums is None:
        raise RuntimeError("Prototype extraction did not produce any aerial features.")

    valid_mask = counts > 0
    prototypes = torch.zeros_like(feature_sums)
    prototypes[valid_mask] = feature_sums[valid_mask] / counts[valid_mask].unsqueeze(1).float()
    prototypes[valid_mask] = torch.nn.functional.normalize(prototypes[valid_mask], dim=1)

    torch.save(
        {
            "prototypes": prototypes,
            "counts": counts,
            "valid_mask": valid_mask,
            "dataset_name": "AERI-PEDES",
            "source_checkpoint": args.finetune,
            "num_train_samples": len(image_pids),
            "num_pids": int(valid_mask.sum().item()),
            "img_size": tuple(args.img_size),
        },
        args.output,
    )

    print(f"Saved {int(valid_mask.sum().item())} aerial prototypes to {args.output}")


if __name__ == "__main__":
    main()
