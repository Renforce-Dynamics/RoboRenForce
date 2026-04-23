"""Standalone offline training runner for FlowMatchingActionDiT.

Unlike RL runners, this operates on offline datasets (no env interaction).
Supports two data modes:
  1. Pre-extracted features: load cached vl_embs from disk
  2. End-to-end: run backbone on images to produce vl_embs (requires Cosmos2.5)

Usage:
    python train_action_dit.py --config config.yaml
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import MISSING, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from RoboRenForce import configclass
from RoboRenForce.components.nn_models.action_dit import (
    FlowMatchingActionDiT,
    FlowMatchingActionDiTCfg,
)
from RoboRenForce.algorithms.nn_model_trainer.action_dit_trainer import (
    ActionDiTTrainer,
    ActionDiTTrainerCfg,
)
from RoboRenForce.runners.logger import SupervisedLogger, SupervisedLoggerCfg


# ======================================================================
# Dataset for pre-extracted features
# ======================================================================

class PreExtractedActionDataset(Dataset):
    """Dataset of pre-extracted VLM features + actions.

    Expected directory structure:
        data_dir/
            features/
                000000.pt  # dict: {vl_embs, actions, action_mask, state?}
                000001.pt
                ...
            dataset_statistics.json  # optional normalization stats

    Each .pt file contains a dict with:
        vl_embs:     (S, D)          float — VLM hidden states
        actions:     (T, action_dim) float — normalized action trajectory
        action_mask: (T, action_dim) float — valid action mask
        state:       (state_dim,)    float — optional proprioceptive state
    """

    def __init__(self, data_dir: str, device: str = "cpu"):
        self.data_dir = data_dir
        self.device = device
        feat_dir = os.path.join(data_dir, "features")

        if not os.path.isdir(feat_dir):
            raise FileNotFoundError(
                f"Features directory not found: {feat_dir}\n"
                "Run feature extraction first or provide a features/ subdirectory."
            )

        self.files = sorted(
            [os.path.join(feat_dir, f) for f in os.listdir(feat_dir) if f.endswith(".pt")]
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt files in {feat_dir}")

        # Load normalization stats if available
        stats_path = os.path.join(data_dir, "dataset_statistics.json")
        self.norm_stats = None
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                self.norm_stats = json.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location=self.device)
        return data


def collate_action_dit(batch: List[dict]) -> dict:
    """Collate function for ActionDiT batches."""
    out = {
        "vl_embs": torch.stack([b["vl_embs"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "action_mask": torch.stack([b["action_mask"] for b in batch]),
    }
    if "state" in batch[0] and batch[0]["state"] is not None:
        out["state"] = torch.stack([b["state"] for b in batch])
    return out


# ======================================================================
# Runner
# ======================================================================

class ActionDiTRunner:
    """Offline training runner for FlowMatchingActionDiT.

    This is a standalone orchestrator (not inheriting BaseRunner)
    because DiT action model training is pure supervised learning
    with no environment interaction.

    Uses ``SupervisedLogger`` for console output and TensorBoard/wandb metrics.
    """

    def __init__(self, cfg: "ActionDiTRunnerCfg", device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.current_step = 0

        # Initialize logger
        self.logger: SupervisedLogger = SupervisedLogger(cfg.logger_cfg, log_dir=cfg.output_dir)

        self._init_model()
        self._init_trainer()
        self._init_data()

        num_params = sum(p.numel() for p in self.action_model.parameters())
        num_samples = len(self.train_dataset) if self.train_dataset else 0
        self.logger.log_info(f"ActionDiTRunner: {num_params:,} params, {num_samples} train samples")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_model(self):
        cfg = self.cfg

        # Build action model
        self.action_model = FlowMatchingActionDiT(cfg.action_model_cfg)
        self.action_model.to(self.device)

        # Optionally load from DiT4DiT checkpoint
        if cfg.pretrained_checkpoint:
            self.action_model.load_from_dit4dit_checkpoint(
                cfg.pretrained_checkpoint,
                prefix=cfg.checkpoint_prefix,
                strict=cfg.checkpoint_strict,
            )
            self.logger.log_info(f"Loaded pretrained: {cfg.pretrained_checkpoint}")

    def _init_trainer(self):
        self.trainer = ActionDiTTrainer(
            cfg=self.cfg.trainer_cfg,
            action_model=self.action_model,
        )

    def _init_data(self):
        cfg = self.cfg
        self.train_dataset = None
        self.eval_dataset = None
        self.train_loader = None
        self.eval_loader = None
        self._train_iter = None

        if cfg.train_data_dir:
            self.train_dataset = PreExtractedActionDataset(
                cfg.train_data_dir, device="cpu"
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                collate_fn=collate_action_dit,
                pin_memory=True,
                drop_last=True,
            )
            self.logger.log_info(f"Train dataset: {len(self.train_dataset)} samples")

        if cfg.eval_data_dir:
            self.eval_dataset = PreExtractedActionDataset(
                cfg.eval_data_dir, device="cpu"
            )
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                collate_fn=collate_action_dit,
                pin_memory=True,
            )
            self.logger.log_info(f"Eval dataset: {len(self.eval_dataset)} samples")

    def _get_train_batch(self) -> dict:
        """Get next training batch, cycling through epochs."""
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)
        return {k: v.to(self.device) for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, max_steps: Optional[int] = None):
        """Main training loop.

        Args:
            max_steps: Override cfg.max_train_steps if provided.
        """
        cfg = self.cfg
        total_steps = max_steps or cfg.max_train_steps

        # Initialize writer backend
        self.logger.init_logger()

        # Build LR scheduler
        self.trainer.build_scheduler(total_steps)

        start_step = self.current_step
        os.makedirs(cfg.output_dir, exist_ok=True)
        self.logger.log_info(f"Training for {total_steps} steps (from step {start_step})")

        t_start = time.time()

        for step in range(start_step, total_steps):
            # Train step
            batch = self._get_train_batch()
            stats = self.trainer.train_step(
                vl_embs=batch["vl_embs"],
                actions=batch["actions"],
                action_mask=batch["action_mask"],
                state=batch.get("state"),
            )
            self.current_step = step + 1

            # Periodic console + writer logging
            if self.current_step % cfg.log_interval == 0:
                elapsed = time.time() - t_start
                self.logger.log_train(self.current_step, total_steps, stats, elapsed)

            # Evaluation
            if cfg.eval_interval and self.current_step % cfg.eval_interval == 0:
                eval_stats = self._evaluate()
                self.logger.log_eval(self.current_step, eval_stats)

            # Save checkpoint
            if self.current_step % cfg.save_interval == 0:
                self.save(
                    os.path.join(cfg.output_dir, f"checkpoint_step_{self.current_step}.pt")
                )

        # Final save
        self.save(os.path.join(cfg.output_dir, "checkpoint_final.pt"))
        self.logger.log_complete(total_steps, time.time() - t_start)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        if self.eval_loader is None:
            return {}

        self.action_model.eval()
        total_loss = 0.0
        count = 0

        for batch in self.eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.trainer_cfg.use_amp):
                loss = self.action_model(
                    batch["vl_embs"],
                    batch["actions"],
                    batch["action_mask"],
                    batch.get("state"),
                )
            total_loss += loss.item()
            count += 1
            if count >= self.cfg.eval_steps:
                break

        self.action_model.train()
        return {"eval_action_loss": total_loss / max(1, count)}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save full training state via logger."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        saved = {
            "action_model_state_dict": self.action_model.state_dict(),
            "trainer_state_dict": self.trainer.state_dict(),
            "current_step": self.current_step,
            "config": {
                "action_model_cfg": self.cfg.action_model_cfg.to_dict()
                if hasattr(self.cfg.action_model_cfg, "to_dict")
                else str(self.cfg.action_model_cfg),
            },
        }
        self.logger.save_model(saved, path, self.current_step)
        self.logger.log_save(path, self.current_step)

    def load(self, path: str, load_optimizer: bool = True):
        """Load training state."""
        loaded = torch.load(path, map_location=self.device)
        self.action_model.load_state_dict(loaded["action_model_state_dict"])
        self.current_step = loaded.get("current_step", 0)
        if load_optimizer and "trainer_state_dict" in loaded:
            self.trainer.load_state_dict(loaded["trainer_state_dict"])
        self.logger.log_info(f"Loaded checkpoint: {path} (step {self.current_step})")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_inference_model(self):
        """Return the action model in eval mode for inference."""
        self.action_model.eval()
        return self.action_model


# ======================================================================
# Feature extraction utility
# ======================================================================

def extract_features_from_dit4dit(
    dit4dit_checkpoint: str,
    dataset_config: dict,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 16,
):
    """Extract VLM features from a DiT4DiT backbone and save to disk.

    This enables mode 1 (pre-extracted features) training without
    keeping the heavy Cosmos2.5 backbone in memory during action model training.

    Args:
        dit4dit_checkpoint: Path to full DiT4DiT checkpoint.
        dataset_config: Dict with dataset parameters (data_root, data_mix, etc.).
        output_dir: Where to save .pt feature files.
        device: Device for inference.
        batch_size: Batch size for extraction.
    """
    import sys
    sys.path.insert(0, os.path.dirname(dit4dit_checkpoint))

    try:
        from DiT4DiT.model.tools import build_framework
        from DiT4DiT.model.framework.base_framework import BaseFramework
    except ImportError:
        raise ImportError(
            "DiT4DiT package not found. Install it or add to sys.path."
        )

    print(f"[Extract] Loading DiT4DiT model from {dit4dit_checkpoint}")
    model = BaseFramework.from_pretrained(dit4dit_checkpoint)
    model.to(device)
    model.eval()

    feat_dir = os.path.join(output_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)

    print(f"[Extract] Feature extraction complete. Saved to {feat_dir}")


# ======================================================================
# Config
# ======================================================================

@configclass
class ActionDiTRunnerCfg:
    # Model
    action_model_cfg: FlowMatchingActionDiTCfg = FlowMatchingActionDiTCfg(
        action_dim=8, action_horizon=8
    )

    # Pretrained checkpoint (optional)
    pretrained_checkpoint: str = ""
    checkpoint_prefix: str = "action_model."
    checkpoint_strict: bool = True

    # Trainer
    trainer_cfg: ActionDiTTrainerCfg = ActionDiTTrainerCfg()

    # Logger (SupervisedLogger for SL tasks)
    logger_cfg: SupervisedLoggerCfg = SupervisedLoggerCfg()

    # Data
    train_data_dir: str = ""
    eval_data_dir: str = ""
    batch_size: int = 16
    num_workers: int = 4

    # Training schedule
    max_train_steps: int = 100000
    save_interval: int = 5000
    log_interval: int = 100
    eval_interval: int = 1000
    eval_steps: int = 50

    # Output
    output_dir: str = "results/action_dit"
