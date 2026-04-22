#!/usr/bin/env python3
"""
VLA SFT — Single GPU Training Script

Fine-tunes a pretrained VLA on task-specific demonstrations.

Usage:
    # Mock mode (no pretrained checkpoint needed):
    python scripts/vla/post_train/train_sft.py --mock --epochs 5

    # With pretrained checkpoint:
    python scripts/vla/post_train/train_sft.py \
        --model_type mlp_baseline \
        --pretrained checkpoints/vla_pretrain/checkpoint_final.pt \
        --data_root /path/to/task_data \
        --epochs 20

    # With KL regularization:
    python scripts/vla/post_train/train_sft.py \
        --mock --epochs 5 --kl_coef 0.01
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/RoboRenForce"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/RRF_models"))

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.prototype.embodied import BasePolicy, ForwardType
from RoboRenForce.prototype.embodied.lerobot.lerobot_dataset import LeRobotDatasetCfg
from RoboRenForce.runners.vla.post_train.sft_runner import VLASFTRunner, VLASFTRunnerCfg


# ===== Mock Policy for testing ===== #

class MockSFTPolicy(BasePolicy):
    """Simple MLP policy for testing SFT pipeline without real VLM."""

    def __init__(self, state_dim: int = 32, action_dim: int = 36, action_horizon: int = 1):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, action_dim * action_horizon),
        )

    def predict_action(self, obs, **kwargs):
        states = obs["states"]
        with torch.no_grad():
            h = self.backbone(states)
            raw = self.action_head(h)
        return raw.view(states.shape[0], self.action_horizon, self.action_dim)

    def forward(self, forward_type=ForwardType.PRETRAIN, **kwargs):
        if forward_type in (ForwardType.PRETRAIN, ForwardType.SFT):
            return self.pretrain_forward(**kwargs)
        elif forward_type == ForwardType.INFERENCE:
            return {"actions": self.predict_action(kwargs["obs"])}
        raise NotImplementedError(f"{forward_type}")

    def pretrain_forward(self, obs, target_actions, **kwargs):
        states = obs["states"]
        h = self.backbone(states)
        raw = self.action_head(h)
        pred = raw.view(states.shape[0], self.action_horizon, self.action_dim)
        loss = nn.functional.mse_loss(pred, target_actions)
        return {"loss": loss, "action_loss": loss, "pred_actions": pred}

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)


class MockSFTDataset(torch.utils.data.Dataset):
    """Mock dataset that produces (states, actions) for SFT testing."""

    def __init__(self, cfg=None, n=300, state_dim=32, action_dim=36):
        self.n = n
        # Generate consistent data so loss can decrease
        torch.manual_seed(0)
        self.states = torch.randn(n, state_dim)
        # Project states to action_dim to handle action_dim > state_dim
        proj = torch.randn(state_dim, action_dim) * 0.1
        self.actions = torch.sin(self.states @ proj) * 0.5

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "states": self.states[idx],
            "action": self.actions[idx],
        }


@configclass
class MockSFTDatasetCfg(ModuleBaseCfg):
    class_type: type = MockSFTDataset


# ===== Mock model registry hook ===== #

_MOCK_REGISTRY = {"mock_sft": MockSFTPolicy}


def _patch_registry():
    """Patch RRF_models.get_model to support mock_sft."""
    try:
        import RRF_models
        _original = RRF_models.get_model

        def patched_get_model(name, cfg=None, **kwargs):
            if name in _MOCK_REGISTRY:
                return _MOCK_REGISTRY[name]()
            return _original(name, cfg=cfg, **kwargs)

        RRF_models.get_model = patched_get_model
    except ImportError:
        # If RRF_models not installed, install minimal mock
        import types
        mod = types.ModuleType("RRF_models")
        mod.get_model = lambda name, cfg=None, **kw: _MOCK_REGISTRY[name]()
        sys.modules["RRF_models"] = mod


# ===== Main ===== #

def main():
    parser = argparse.ArgumentParser(description="VLA SFT (Single-GPU)")
    parser.add_argument("--mock", action="store_true", help="Use mock policy + mock data")
    parser.add_argument("--model_type", type=str, default="mlp_baseline")
    parser.add_argument("--pretrained", type=str, default="", help="Pretrained checkpoint path")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--kl_coef", type=float, default=0.0)
    parser.add_argument("--action_loss_type", type=str, default="mse", choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--scheduler", type=str, default="warmup", choices=["warmup", "cosine"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/sft/")
    parser.add_argument("--log_dir", type=str, default="logs/sft/")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mock:
        _patch_registry()
        cfg = VLASFTRunnerCfg(
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
        cfg = VLASFTRunnerCfg(
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

    print(f"Device: {args.device}")
    print(f"Mode: {'mock' if args.mock else 'real'}")
    print(f"KL coef: {args.kl_coef}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    runner = VLASFTRunner(cfg, log_dir=args.log_dir, device=args.device)
    runner.learn()


if __name__ == "__main__":
    main()
