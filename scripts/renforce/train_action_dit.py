"""Train FlowMatchingActionDiT — offline action model training.

Usage:
    # With pre-extracted features:
    python train_action_dit.py --train-data-dir /path/to/features --action-dim 8

    # With DiT4DiT pretrained checkpoint (fine-tune):
    python train_action_dit.py --train-data-dir /path/to/features \
        --pretrained /path/to/steps_N_pytorch_model.pt

    # Extract features first (requires DiT4DiT in path):
    python train_action_dit.py --extract-features \
        --dit4dit-checkpoint /path/to/checkpoint \
        --dataset-root /path/to/libero \
        --output-dir /path/to/features

    # Generate dummy data for testing:
    python train_action_dit.py --generate-dummy --train-data-dir /tmp/dummy_action_dit
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/RoboRenForce"))

from RoboRenForce.components.nn_models.action_dit import (
    FlowMatchingActionDiTCfg,
)
from RoboRenForce.algorithms.nn_model_trainer.action_dit_trainer import (
    ActionDiTTrainerCfg,
)
from RoboRenForce.runners.nn_model_based.action_dit_runner import (
    ActionDiTRunner,
    ActionDiTRunnerCfg,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_dummy_data(data_dir: str, num_samples: int = 200, cfg=None):
    """Generate dummy data for testing the training pipeline."""
    feat_dir = os.path.join(data_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)

    action_dim = cfg.action_dim if cfg else 8
    state_dim = cfg.state_dim if cfg else 16
    action_horizon = cfg.action_horizon if cfg else 8
    vl_dim = cfg.dit_cross_attention_dim if cfg else 2048
    vl_seq_len = 32

    for i in range(num_samples):
        sample = {
            "vl_embs": torch.randn(vl_seq_len, vl_dim),
            "actions": torch.randn(action_horizon, action_dim),
            "action_mask": torch.ones(action_horizon, action_dim),
        }
        if state_dim > 0:
            sample["state"] = torch.randn(state_dim)
        torch.save(sample, os.path.join(feat_dir, f"{i:06d}.pt"))

    logger.info("Generated %d dummy samples in %s", num_samples, feat_dir)


def main():
    parser = argparse.ArgumentParser(description="Train FlowMatchingActionDiT")

    # Data
    parser.add_argument("--train-data-dir", type=str, default="")
    parser.add_argument("--eval-data-dir", type=str, default="")

    # Model
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--dit-variant", type=str, default="DiT-B", choices=["DiT-B", "DiT-L"])
    parser.add_argument("--dit-num-layers", type=int, default=16)

    # Pretrained
    parser.add_argument("--pretrained", type=str, default="")

    # Training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--repeated-diffusion-steps", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="results/action_dit")

    # Utilities
    parser.add_argument("--generate-dummy", action="store_true",
                        help="Generate dummy data for testing")
    parser.add_argument("--dummy-samples", type=int, default=200)
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint path")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Build configs
    model_cfg = FlowMatchingActionDiTCfg(
        dit_variant=args.dit_variant,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        action_horizon=args.action_horizon,
        dit_num_layers=args.dit_num_layers,
        dit_cross_attention_dim=2048,
        dit_output_dim=2560,
        dit_dropout=0.2,
        dit_final_dropout=True,
        dit_interleave_self_attention=True,
        hidden_size=2560,
        use_amp=True,
    )

    # Generate dummy data if requested
    if args.generate_dummy:
        generate_dummy_data(args.train_data_dir, args.dummy_samples, model_cfg)
        if not args.max_steps:
            return

    trainer_cfg = ActionDiTTrainerCfg(
        learning_rate=args.lr,
        num_warmup_steps=args.warmup_steps,
        repeated_diffusion_steps=args.repeated_diffusion_steps,
        use_amp=True,
    )

    runner_cfg = ActionDiTRunnerCfg(
        action_model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        pretrained_checkpoint=args.pretrained,
        train_data_dir=args.train_data_dir,
        eval_data_dir=args.eval_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_steps=args.max_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        output_dir=args.output_dir,
    )

    # Build runner
    runner = ActionDiTRunner(runner_cfg, device=args.device)

    # Resume if specified
    if args.resume:
        runner.load(args.resume)

    # Train
    runner.learn()


if __name__ == "__main__":
    main()
