#!/usr/bin/env python3
"""
VLA SFT — Multi-GPU DDP Training Script

Usage:
    # 2 GPUs, mock mode:
    torchrun --nproc_per_node=2 scripts/vla/post_train/train_sft_ddp.py --mock --epochs 5

    # 8 GPUs with real data:
    torchrun --nproc_per_node=8 scripts/vla/post_train/train_sft_ddp.py \
        --model_type mlp_baseline \
        --pretrained checkpoints/vla_pretrain/checkpoint_final.pt \
        --data_root /path/to/task_data \
        --epochs 20 --kl_coef 0.01
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/RoboRenForce"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/RRF_models"))

import torch

from RoboRenForce.prototype.embodied.lerobot.lerobot_dataset import LeRobotDatasetCfg
from RoboRenForce.runners.vla.post_train.sft_runner_distributed import (
    DistributedVLASFTRunner,
    DistributedVLASFTRunnerCfg,
)

# Reuse mock classes from single-GPU script
from train_sft import MockSFTDatasetCfg, _patch_registry


def main():
    parser = argparse.ArgumentParser(description="VLA SFT (Multi-GPU DDP)")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model_type", type=str, default="mlp_baseline")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--kl_coef", type=float, default=0.0)
    parser.add_argument("--action_loss_type", type=str, default="mse")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--scheduler", type=str, default="warmup")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/sft_ddp/")
    parser.add_argument("--log_dir", type=str, default="logs/sft_ddp/")
    args = parser.parse_args()

    if args.mock:
        _patch_registry()
        cfg = DistributedVLASFTRunnerCfg(
            model_type="mock_sft",
            dataset_cfg=MockSFTDatasetCfg(),
            pretrained_checkpoint="",
            freeze_backbone=args.freeze_backbone,
            kl_coef=args.kl_coef,
            action_loss_type=args.action_loss_type,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            scheduler_type=args.scheduler,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            num_workers=0,
            log_interval=10,
            save_interval=args.save_interval,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        dataset_cfg = LeRobotDatasetCfg(data_root=args.data_root) if args.data_root else MockSFTDatasetCfg()
        cfg = DistributedVLASFTRunnerCfg(
            model_type=args.model_type,
            pretrained_checkpoint=args.pretrained,
            dataset_cfg=dataset_cfg,
            freeze_backbone=args.freeze_backbone,
            kl_coef=args.kl_coef,
            action_loss_type=args.action_loss_type,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            scheduler_type=args.scheduler,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            num_workers=args.num_workers,
            log_interval=10,
            save_interval=args.save_interval,
            checkpoint_dir=args.checkpoint_dir,
        )

    runner = DistributedVLASFTRunner(cfg, log_dir=args.log_dir)
    runner.learn()


if __name__ == "__main__":
    main()
