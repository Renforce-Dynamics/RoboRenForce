#!/usr/bin/env python3
"""
VLA Pretrain — Single GPU Training Script

Usage:
    # With mock VLM (no GPU required, for testing):
    python scripts/vla/pretrain/train_single_gpu.py --mock --epochs 5

    # With real Qwen2-VL (requires GPU + model download):
    python scripts/vla/pretrain/train_single_gpu.py \
        --data_root data/dummy_dataset \
        --model_name Qwen/Qwen2-VL-2B-Instruct \
        --epochs 10 \
        --batch_size 4

    # Resume from checkpoint:
    python scripts/vla/pretrain/train_single_gpu.py \
        --resume checkpoints/checkpoint_step_1000.pt
"""

import argparse
import sys
import os

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/RoboRenForce"))

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.vlm.vlm_backbone_base import VLMBackbone, VLMBackboneCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import RegressionActionHeadCfg
from RoboRenForce.components.actor.action_heads.diffusion_action_head import DiffusionActionHeadCfg
from RoboRenForce.components.actor.vla_actor import VLAActorCfg
from RoboRenForce.algorithms.vla_training.pretrain_algorithm import VLAPretrainAlgorithmCfg
from RoboRenForce.dataset.lerobot.lerobot_dataset import LeRobotDatasetCfg
from RoboRenForce.runners.vla.pretrain.vla_pretrain_runner import (
    VLAPretrainRunner,
    VLAPretrainRunnerCfg,
)


# ===== Mock VLM for testing without real model ===== #

class MockVLM(VLMBackbone):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, cfg.output_dim),
        )
        self.output_dim = cfg.output_dim

    def forward(self, image, text=None, **kwargs):
        return self.encoder(image)


@configclass
class MockVLMCfg(VLMBackboneCfg):
    class_type: type = MockVLM
    model_name: str = "mock"
    output_dim: int = 64
    freeze: bool = True


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, cfg=None):
        self.n = 200

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 64, 64),
            "proprioception": torch.randn(12),
            "action": torch.randn(7),
        }


@configclass
class MockDatasetCfg(ModuleBaseCfg):
    class_type: type = MockDataset


# ===== Config builders ===== #

def build_psi0_mock_config(args) -> VLAPretrainRunnerCfg:
    """Use MockVLM with real Psi0 data (for testing pipeline without loading real VLM)."""
    action_head_cfg = RegressionActionHeadCfg(
        action_dim=args.action_dim, action_horizon=1, hidden_dims=[256, 256],
    ) if args.head == "regression" else DiffusionActionHeadCfg(
        action_dim=args.action_dim, action_horizon=1,
        num_layers=4, num_heads=8, embed_dim=256,
        num_train_steps=100, num_diffusion_steps=10,
    )

    return VLAPretrainRunnerCfg(
        vla_actor_cfg=VLAActorCfg(
            vlm_backbone_cfg=MockVLMCfg(output_dim=256),
            freeze_vlm=True,
            fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
            action_head_cfg=action_head_cfg,
            use_proprioception=True,
        ),
        algorithm_cfg=VLAPretrainAlgorithmCfg(
            learning_rate=args.lr, warmup_steps=100, use_amp=args.amp,
        ),
        dataset_cfg=LeRobotDatasetCfg(
            data_root=args.data_root,
            load_videos=True,
            frames_dir=args.frames_dir,
            image_size=tuple(args.image_size),
        ),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        log_interval=10,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )


def build_mock_config(args) -> VLAPretrainRunnerCfg:
    action_head_cfg = RegressionActionHeadCfg(
        action_dim=7, action_horizon=1, hidden_dims=[128, 128],
    ) if args.head == "regression" else DiffusionActionHeadCfg(
        action_dim=7, action_horizon=1,
        num_layers=2, num_heads=4, embed_dim=64,
        num_train_steps=50, num_diffusion_steps=10,
    )

    return VLAPretrainRunnerCfg(
        vla_actor_cfg=VLAActorCfg(
            vlm_backbone_cfg=MockVLMCfg(output_dim=64),
            freeze_vlm=True,
            fusion_cfg=FusionLayerCfg(output_dim=128, hidden_dims=[128]),
            action_head_cfg=action_head_cfg,
            use_proprioception=True,
        ),
        algorithm_cfg=VLAPretrainAlgorithmCfg(
            learning_rate=args.lr, warmup_steps=100, use_amp=False,
        ),
        dataset_cfg=MockDatasetCfg(),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=0,
        log_interval=10,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )


def build_qwen2vl_config(args) -> VLAPretrainRunnerCfg:
    from RoboRenForce.networks.vlm.qwen2vl import Qwen2VLCfg

    action_head_cfg = RegressionActionHeadCfg(
        action_dim=args.action_dim, action_horizon=1, hidden_dims=[256, 256],
    ) if args.head == "regression" else DiffusionActionHeadCfg(
        action_dim=args.action_dim, action_horizon=1,
        num_layers=4, num_heads=8, embed_dim=256,
        num_train_steps=100, num_diffusion_steps=10,
    )

    return VLAPretrainRunnerCfg(
        vla_actor_cfg=VLAActorCfg(
            vlm_backbone_cfg=Qwen2VLCfg(model_name=args.model_name, freeze=True, device_map_auto=False),
            freeze_vlm=True,
            fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
            action_head_cfg=action_head_cfg,
            use_proprioception=True,
        ),
        algorithm_cfg=VLAPretrainAlgorithmCfg(
            learning_rate=args.lr, warmup_steps=1000, use_amp=args.amp,
        ),
        dataset_cfg=LeRobotDatasetCfg(
            data_root=args.data_root,
            load_videos=True,
            frames_dir=args.frames_dir,
            image_size=tuple(args.image_size),
        ),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        log_interval=10,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="VLA Pretrain (Single-GPU)")
    parser.add_argument("--mock", action="store_true", help="Use mock VLM + mock data for CPU testing")
    parser.add_argument("--psi0", action="store_true", help="Use mock VLM + real Psi0 data")
    parser.add_argument("--data_root", type=str, default="data/example_dataset",
                        help="Path to LeRobot-format dataset directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--head", type=str, default="regression", choices=["regression", "diffusion"])
    parser.add_argument("--action_dim", type=int, default=36)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--frames_dir", type=str, default="", help="Pre-extracted frames directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/vla_pretrain/")
    parser.add_argument("--log_dir", type=str, default="logs/vla_pretrain/")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {args.device}")
    print(f"Mode: {'mock' if args.mock else 'real'}")
    print(f"Action head: {args.head}")

    if args.mock:
        cfg = build_mock_config(args)
    elif args.psi0:
        cfg = build_psi0_mock_config(args)
    else:
        cfg = build_qwen2vl_config(args)

    runner = VLAPretrainRunner(cfg, log_dir=args.log_dir, device=args.device)

    if args.resume:
        runner.load_checkpoint(args.resume)

    runner.learn(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
