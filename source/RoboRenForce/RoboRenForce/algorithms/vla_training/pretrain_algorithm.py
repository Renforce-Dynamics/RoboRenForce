"""
VLA Pretraining Algorithm

Supervised learning algorithm for VLA pretraining on offline demonstrations.

IO Contract:
    compute_loss(batch, vla_actor):
        Input:
            batch: {"image": (B,C,H,W), "proprioception": (B,D), "action": (B,H,A)}
            vla_actor: VLAActor module
        Output:
            loss_dict: {"total_loss": scalar, "action_loss": scalar, ...}

    update(batch, vla_actor, optimizer):
        Input:  same as compute_loss + optimizer
        Output: loss_dict with scalar values for logging
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


class VLAPretrainAlgorithm(ModuleBase):
    """
    Supervised pretraining algorithm for VLA.

    Supports:
    - Regression loss (MSE/L1) for RegressionActionHead
    - Diffusion loss (noise prediction MSE) for DiffusionActionHead
    - Mixed precision training
    - Gradient clipping
    """

    def __init__(self, cfg: VLAPretrainAlgorithmCfg):
        super().__init__()
        self.cfg = cfg

        self.use_amp = cfg.use_amp
        if cfg.use_amp:
            self.amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
            self.scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp_dtype == "fp16"))
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

        self._step = 0

    def compute_loss(self, batch: dict, vla_actor: nn.Module) -> dict:
        obs_dict = {"image": batch["image"]}
        if "proprioception" in batch:
            obs_dict["proprioception"] = batch["proprioception"]
        if "text" in batch:
            obs_dict["text"] = batch["text"]

        target_actions = batch["action"]
        if target_actions.ndim == 2:
            target_actions = target_actions.unsqueeze(1)

        # Use forward() with target_actions for DDP compatibility
        train_output = vla_actor(obs_dict, target_actions=target_actions)

        if "noise_pred" in train_output:
            action_loss = F.mse_loss(train_output["noise_pred"], train_output["noise_target"])
        else:
            pred = train_output["pred_actions"]
            target = train_output["target_actions"]
            if self.cfg.action_loss_type == "l1":
                action_loss = F.l1_loss(pred, target)
            else:
                action_loss = F.mse_loss(pred, target)

        total_loss = self.cfg.action_loss_weight * action_loss

        return {"total_loss": total_loss, "action_loss": action_loss.detach()}

    def update(self, batch: dict, vla_actor: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
        loss_dict = self.compute_loss(batch, vla_actor)
        total_loss = loss_dict["total_loss"]

        optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vla_actor.parameters(), self.cfg.max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vla_actor.parameters(), self.cfg.max_grad_norm)
            optimizer.step()

        self._step += 1

        return {k: v.detach().item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()}

    def get_lr_scale(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return step / max(1, self.cfg.warmup_steps)
        return 1.0


@configclass
class VLAPretrainAlgorithmCfg(ModuleBaseCfg):
    """VLA pretraining algorithm configuration."""

    class_type: type[VLAPretrainAlgorithm] = VLAPretrainAlgorithm

    action_loss_type: str = "mse"
    action_loss_weight: float = 1.0

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    use_amp: bool = False
    amp_dtype: str = "bf16"
