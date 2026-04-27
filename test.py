import argparse
import os.path as op

from datasets import build_dataloader
from model.build_finetune import build_finetune_model
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AERI baseline test")
    parser.add_argument("--config_file", required=True, help="Path to saved configs.yaml")
    parser.add_argument("--checkpoint", default="", help="Optional explicit checkpoint path")
    parser.add_argument("--root_dir", default="", help="Override dataset root_dir from config")
    parser.add_argument("--output_dir", default="", help="Override output_dir from config for local logs")
    args_cli = parser.parse_args()

    args = load_train_configs(args_cli.config_file)
    args.training = False
    args.loss_names = "base"
    if args_cli.root_dir:
        args.root_dir = args_cli.root_dir
    if args_cli.output_dir:
        args.output_dir = args_cli.output_dir

    checkpoint_path = args_cli.checkpoint
    if not checkpoint_path:
        best_ckpt = op.join(args.output_dir, "best0.pth")
        final_ckpt = op.join(args.output_dir, "final.pth")
        checkpoint_path = best_ckpt if op.exists(best_ckpt) else final_ckpt

    logger = setup_logger("IRRA", save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_finetune_model(args, num_classes=num_classes)
    Checkpointer(model).load(f=checkpoint_path)
    model.to("cuda")
    do_inference(model, test_img_loader, test_txt_loader)
