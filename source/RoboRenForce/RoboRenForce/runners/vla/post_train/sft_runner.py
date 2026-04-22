"""
VLA SFT Runner (Single-GPU)

Runner for supervised fine-tuning of a pretrained VLA on task-specific demonstrations.
Loads a pretrained checkpoint, optionally applies LoRA, and fine-tunes with optional
KL regularization against the pretrained reference policy.

IO Contract:
    __init__(cfg, log_dir, device):
        Loads pretrained checkpoint, sets up reference policy, configures LoRA.

    learn(num_epochs):
        Fine-tuning loop with KL regularization and validation.

Reference: TODO.md Phase 5, .references/Psi0/src/psi/trainers/pretrain.py
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.algorithms.vla_training.sft import SFTAlgorithmCfg
from RoboRenForce.prototype.embodied import BasePolicy, ForwardType


class VLASFTRunner(ModuleBase):
    """
    Single-GPU VLA SFT runner.

    Pipeline:
    1. Construct policy (via model registry or direct cfg)
    2. Load pretrained checkpoint into policy
    3. Optionally create frozen reference policy for KL regularization
    4. Optionally apply LoRA to action head / backbone
    5. Fine-tune with SFT algorithm on task-specific data
    """

    def __init__(self, cfg: "VLASFTRunnerCfg", log_dir: str = "logs/", device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load dataset
        self.dataset = cfg.dataset_cfg.construct_from_cfg()

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

        # 2. Construct policy
        if cfg.model_type is not None:
            from RRF_models import get_model
            self.policy = get_model(cfg.model_type, cfg=cfg.model_cfg)
        else:
            raise ValueError("model_type must be specified for SFT runner")
        self.policy.to(device)

        # 3. Load pretrained checkpoint
        if cfg.pretrained_checkpoint:
            self._load_pretrained(cfg.pretrained_checkpoint)

        # 4. Freeze backbone if requested
        if cfg.freeze_backbone:
            self.policy.freeze_backbone()
            print(f"Backbone frozen. Trainable params: {self.policy.num_trainable_params():,}")

        # 5. Create reference policy for KL regularization
        self.ref_policy: Optional[BasePolicy] = None
        if cfg.kl_coef > 0:
            self.ref_policy = copy.deepcopy(self.policy)
            self.ref_policy.to(device)
            for p in self.ref_policy.parameters():
                p.requires_grad_(False)
            self.ref_policy.eval()
            print("Reference policy created for KL regularization")

        # 6. Algorithm
        algo_cfg = SFTAlgorithmCfg(
            action_loss_type=cfg.action_loss_type,
            action_loss_weight=cfg.action_loss_weight,
            kl_coef=cfg.kl_coef,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            warmup_steps=cfg.warmup_steps,
        )
        self.algorithm = algo_cfg.construct_from_cfg()
        if self.ref_policy is not None:
            self.algorithm.set_reference_policy(self.ref_policy)

        # 7. Optimizer (only trainable params)
        trainable_params = list(self.policy.trainable_parameters())
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters! Check freeze_backbone / LoRA settings.")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # 8. Scheduler
        if cfg.scheduler_type == "cosine":
            total_steps = len(self.train_loader) * cfg.num_epochs
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=cfg.min_lr)
        else:
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(1.0, step / max(1, cfg.warmup_steps)),
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

    def _load_pretrained(self, checkpoint_path: str):
        """Load pretrained VLA checkpoint into policy."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "vla_actor_state_dict" in ckpt:
            state_dict = ckpt["vla_actor_state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        missing, unexpected = self.policy.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained checkpoint: {checkpoint_path}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    def learn(self, num_epochs: Optional[int] = None):
        num_epochs = num_epochs or self.cfg.num_epochs

        n_trainable = self.policy.num_trainable_params()
        n_total = self.policy.num_total_params()
        print(f"Starting SFT: {num_epochs} epochs, "
              f"{len(self.train_loader)} batches/epoch, device={self.device}")
        print(f"  Trainable: {n_trainable:,} / {n_total:,} params "
              f"({100*n_trainable/max(1,n_total):.1f}%)")
        if self.cfg.kl_coef > 0:
            print(f"  KL coef: {self.cfg.kl_coef}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.policy.train()
            epoch_loss = 0.0
            epoch_action_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_steps = 0
            t_start = time.time()

            for batch in self.train_loader:
                batch = self._to_device(batch)

                loss_dict = self.algorithm.update(batch, self.policy, self.optimizer)
                self.scheduler.step()

                epoch_loss += loss_dict["total_loss"]
                epoch_action_loss += loss_dict.get("action_loss", 0)
                epoch_kl_loss += loss_dict.get("kl_loss", 0)
                epoch_steps += 1
                self.global_step += 1

                if self.global_step % self.cfg.log_interval == 0:
                    self.writer.add_scalar("sft/total_loss", loss_dict["total_loss"], self.global_step)
                    self.writer.add_scalar("sft/action_loss", loss_dict.get("action_loss", 0), self.global_step)
                    if "kl_loss" in loss_dict:
                        self.writer.add_scalar("sft/kl_loss", loss_dict["kl_loss"], self.global_step)
                    self.writer.add_scalar("sft/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

                if epoch_steps % 100 == 0:
                    elapsed = time.time() - t_start
                    kl_str = f" | kl={loss_dict.get('kl_loss', 0):.4f}" if "kl_loss" in loss_dict else ""
                    print(f"  step {epoch_steps}/{len(self.train_loader)} | "
                          f"loss={loss_dict['total_loss']:.4f} | "
                          f"action={loss_dict.get('action_loss', 0):.4f}"
                          f"{kl_str} | time={elapsed:.1f}s", flush=True)

                if self.cfg.save_interval > 0 and self.global_step % self.cfg.save_interval == 0:
                    self.save_checkpoint()

            avg_loss = epoch_loss / max(1, epoch_steps)
            avg_action = epoch_action_loss / max(1, epoch_steps)
            elapsed = time.time() - t_start
            print(f"Epoch {epoch+1}/{num_epochs} | loss={avg_loss:.4f} | "
                  f"action_loss={avg_action:.4f} | time={elapsed:.1f}s")

            val_loss = self.validate()
            self.writer.add_scalar("sft/val_loss", val_loss, self.global_step)
            print(f"  val_loss={val_loss:.4f}")

        self.save_checkpoint(tag="final")
        print("SFT training complete.")

    @torch.no_grad()
    def validate(self) -> float:
        self.policy.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            loss_dict = self.algorithm.compute_loss(batch, self.policy)
            total_loss += loss_dict["total_loss"].item()
            n_batches += 1

        self.policy.train()
        return total_loss / max(1, n_batches)

    def save_checkpoint(self, tag: Optional[str] = None):
        tag = tag or f"step_{self.global_step}"
        path = self.checkpoint_dir / f"sft_checkpoint_{tag}.pt"
        torch.save({
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.current_epoch = ckpt.get("epoch", 0)
        print(f"Checkpoint loaded: step={self.global_step}, epoch={self.current_epoch}")

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}


class _DummyWriter:
    def add_scalar(self, *args, **kwargs): pass
    def close(self): pass


@configclass
class VLASFTRunnerCfg(ModuleBaseCfg):
    """SFT runner configuration."""

    class_type: type[VLASFTRunner] = VLASFTRunner

    # Model
    model_type: str = None
    model_cfg: object = None

    # Pretrained checkpoint
    pretrained_checkpoint: str = ""

    # Backbone control
    freeze_backbone: bool = True

    # Dataset
    dataset_cfg: object = None

    # SFT-specific
    action_loss_type: str = "mse"
    action_loss_weight: float = 1.0
    kl_coef: float = 0.0              # >0 enables KL regularization

    # Training
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    scheduler_type: str = "warmup"    # "warmup" or "cosine"

    batch_size: int = 32
    num_epochs: int = 10
    num_workers: int = 4
    val_split: float = 0.1

    save_interval: int = 500
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints/sft/"
