"""
VLA Pretraining Runner (Single-GPU)

Runner for single-GPU VLA pretraining on offline demonstrations.

IO Contract:
    __init__(cfg, log_dir, device):
        Constructs dataset, VLA actor, algorithm, optimizer, scheduler.

    learn(num_epochs):
        Runs training loop. Logs metrics. Saves checkpoints.

    validate():
        Runs eval pass on validation data, returns avg loss.

    save_checkpoint(path) / load_checkpoint(path):
        Saves/loads full training state.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.components.actor.vla_actor import VLAActorCfg
from RoboRenForce.algorithms.vla_training.pretrain_algorithm import VLAPretrainAlgorithmCfg
from RoboRenForce.prototype.embodied.lerobot.lerobot_dataset import LeRobotDatasetCfg


class VLAPretrainRunner(ModuleBase):
    """
    Single-GPU VLA pretraining runner.

    Pipeline:
    1. Load dataset (LeRobot format) -> train/val split
    2. Create VLA actor (VLM + fusion + action head)
    3. Train with supervised loss (MSE or diffusion)
    4. Save checkpoints periodically
    """

    def __init__(self, cfg: VLAPretrainRunnerCfg, log_dir: str = "logs/", device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load dataset
        self.dataset = cfg.dataset_cfg.construct_from_cfg()

        # Train/val split
        n_total = len(self.dataset)
        n_val = max(1, int(n_total * cfg.val_split))
        n_train = n_total - n_val
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device != "cpu"),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # 2. Infer dimensions and construct VLA actor
        dim_params = self._get_dim_params()
        self.vla_actor = cfg.vla_actor_cfg.construct_from_cfg(dim_params)
        self.vla_actor.to(device)

        # 3. Algorithm
        self.algorithm = cfg.algorithm_cfg.construct_from_cfg()

        # 4. Optimizer (only trainable params)
        trainable_params = [p for p in self.vla_actor.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.algorithm.cfg.learning_rate,
            weight_decay=self.algorithm.cfg.weight_decay,
        )

        # 5. Scheduler
        warmup_steps = self.algorithm.cfg.warmup_steps
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / max(1, warmup_steps)),
        )

        self.global_step = 0
        self.current_epoch = 0
        self._writer = None

    @property
    def writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                self._writer = _DummyWriter()
        return self._writer

    def learn(self, num_epochs: Optional[int] = None):
        num_epochs = num_epochs or self.cfg.num_epochs

        self.vla_actor.train()
        print(f"Starting training: {num_epochs} epochs, "
              f"{len(self.train_loader)} batches/epoch, device={self.device}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0
            t_start = time.time()

            for batch in self.train_loader:
                batch = self._to_device(batch)

                loss_dict = self.algorithm.update(batch, self.vla_actor, self.optimizer)
                self.scheduler.step()

                epoch_loss += loss_dict["total_loss"]
                epoch_steps += 1
                self.global_step += 1

                if self.global_step % self.cfg.log_interval == 0:
                    self.writer.add_scalar("train/total_loss", loss_dict["total_loss"], self.global_step)
                    self.writer.add_scalar("train/action_loss", loss_dict["action_loss"], self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

                if epoch_steps % 100 == 0:
                    elapsed_so_far = time.time() - t_start
                    print(f"  step {epoch_steps}/{len(self.train_loader)} | "
                          f"loss={loss_dict['total_loss']:.4f} | "
                          f"time={elapsed_so_far:.1f}s", flush=True)

                if self.cfg.save_interval > 0 and self.global_step % self.cfg.save_interval == 0:
                    self.save_checkpoint()

            avg_loss = epoch_loss / max(1, epoch_steps)
            elapsed = time.time() - t_start
            print(f"Epoch {epoch+1}/{num_epochs} | loss={avg_loss:.4f} | "
                  f"steps={epoch_steps} | time={elapsed:.1f}s")

            val_loss = self.validate()
            self.writer.add_scalar("val/total_loss", val_loss, self.global_step)
            print(f"  val_loss={val_loss:.4f}")

        self.save_checkpoint(tag="final")
        print("Training complete.")

    @torch.no_grad()
    def validate(self) -> float:
        self.vla_actor.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            loss_dict = self.algorithm.compute_loss(batch, self.vla_actor)
            total_loss += loss_dict["total_loss"].item()
            n_batches += 1

        self.vla_actor.train()
        return total_loss / max(1, n_batches)

    def save_checkpoint(self, tag: Optional[str] = None):
        tag = tag or f"step_{self.global_step}"
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "vla_actor_state_dict": self.vla_actor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.vla_actor.load_state_dict(ckpt["vla_actor_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.current_epoch = ckpt.get("epoch", 0)
        print(f"Checkpoint loaded: step={self.global_step}, epoch={self.current_epoch}")

    def _get_dim_params(self) -> dict:
        sample = self.dataset[0]
        dim_params = {}

        if "action" in sample:
            action = sample["action"]
            dim_params["action_dim"] = action.shape[-1] if hasattr(action, "shape") else len(action)

        for key in ("observation.state", "proprioception"):
            if key in sample:
                state = sample[key]
                dim_params["proprioception_dim"] = state.shape[-1] if hasattr(state, "shape") else len(state)
                break

        return dim_params

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}


class _DummyWriter:
    def add_scalar(self, *args, **kwargs): pass
    def close(self): pass


@configclass
class VLAPretrainRunnerCfg(ModuleBaseCfg):
    """Single-GPU VLA pretraining runner configuration."""

    class_type: type[VLAPretrainRunner] = VLAPretrainRunner

    vla_actor_cfg: VLAActorCfg = None
    algorithm_cfg: VLAPretrainAlgorithmCfg = VLAPretrainAlgorithmCfg()
    dataset_cfg: object = None

    batch_size: int = 32
    num_epochs: int = 10
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    val_split: float = 0.1

    save_interval: int = 1000
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints/"
