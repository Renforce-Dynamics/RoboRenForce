"""
SFT — Supervised Fine-Tuning for VLA policies.

Distinct from pretrain in that SFT includes optional KL regularization
against a reference policy, and is designed for fine-tuning an already
pretrained model on task-specific demonstrations.

Algorithm:
    1. Forward pass: compute action predictions from obs + target actions
    2. Action loss: MSE or L1 between predicted and target actions
    3. Optional KL penalty: regularize against reference policy's logprobs
    4. Update with gradient clipping

Reference: RLinf rlinf/workers/sft/fsdp_sft_worker.py
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.prototype.embodied import BasePolicy, ForwardType


class SFTAlgorithm(ModuleBase):
    """Supervised Fine-Tuning algorithm for VLA policies.

    Supports:
    - MSE / L1 / smooth_l1 action loss
    - Optional KL penalty against reference policy
    - Gradient clipping
    - Mixed precision (external scaler)

    Usage:
        sft = SFTAlgorithm(cfg)
        metrics = sft.update(batch, policy, optimizer)
    """

    def __init__(self, cfg: "SFTAlgorithmCfg"):
        super().__init__()
        self.cfg = cfg
        self._step = 0
        self._ref_policy: Optional[BasePolicy] = None

    def set_reference_policy(self, ref_policy: BasePolicy):
        """Set reference policy for KL regularization.

        The reference policy should be frozen (no gradients).
        """
        self._ref_policy = ref_policy
        for p in ref_policy.parameters():
            p.requires_grad_(False)

    def compute_action_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute action prediction loss.

        Args:
            pred_actions: [B, horizon, action_dim] or [B, action_dim]
            target_actions: same shape as pred_actions
        """
        if self.cfg.action_loss_type == "mse":
            return F.mse_loss(pred_actions, target_actions)
        elif self.cfg.action_loss_type == "l1":
            return F.l1_loss(pred_actions, target_actions)
        elif self.cfg.action_loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_actions, target_actions)
        else:
            raise ValueError(f"Unknown action_loss_type: {self.cfg.action_loss_type}")

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        policy: BasePolicy,
    ) -> dict[str, torch.Tensor]:
        """Compute SFT loss.

        Args:
            batch: dict with obs keys + "action" target
            policy: the policy to train

        Returns:
            dict with total_loss, action_loss, optional kl_loss
        """
        # Map batch to standardized obs
        obs = {}
        if "main_images" in batch:
            obs["main_images"] = batch["main_images"]
        if "states" in batch:
            obs["states"] = batch["states"]
        if "task_descriptions" in batch:
            obs["task_descriptions"] = batch["task_descriptions"]

        target_actions = batch["action"]
        if target_actions.dim() == 2:
            target_actions = target_actions.unsqueeze(1)  # [B, 1, action_dim]

        # Forward through policy (handle DDP wrapper)
        raw = policy.module if hasattr(policy, "module") else policy
        fwd_type = ForwardType.SFT if hasattr(raw, "sft_forward") else ForwardType.PRETRAIN
        result = policy.forward(
            fwd_type,
            obs=obs,
            target_actions=target_actions,
        )

        # If policy returns a loss directly, use it
        if "loss" in result:
            action_loss = result["loss"]
        elif "pred_actions" in result:
            action_loss = self.compute_action_loss(result["pred_actions"], target_actions)
        else:
            raise ValueError("Policy must return 'loss' or 'pred_actions'")

        total_loss = self.cfg.action_loss_weight * action_loss
        loss_dict = {"action_loss": action_loss.detach()}

        # KL regularization against reference policy
        if self._ref_policy is not None and self.cfg.kl_coef > 0:
            with torch.no_grad():
                ref_result = self._ref_policy.forward(
                    ForwardType.PRETRAIN, obs=obs, target_actions=target_actions
                )

            if "logprobs" in result and "logprobs" in ref_result:
                kl_div = (result["logprobs"] - ref_result["logprobs"]).mean()
                kl_loss = self.cfg.kl_coef * kl_div
                total_loss = total_loss + kl_loss
                loss_dict["kl_loss"] = kl_loss.detach()
                loss_dict["kl_div"] = kl_div.detach()
            elif "loss" in result and "loss" in ref_result:
                # Approximate KL from loss difference
                kl_approx = (result["loss"] - ref_result["loss"]).abs()
                kl_loss = self.cfg.kl_coef * kl_approx
                total_loss = total_loss + kl_loss
                loss_dict["kl_loss"] = kl_loss.detach()

        loss_dict["total_loss"] = total_loss
        return loss_dict

    def update(
        self,
        batch: dict[str, torch.Tensor],
        policy: BasePolicy,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Compute loss, backprop, step optimizer.

        Returns:
            dict with scalar metrics
        """
        loss_dict = self.compute_loss(batch, policy)
        total_loss = loss_dict["total_loss"]

        optimizer.zero_grad()
        total_loss.backward()
        if self.cfg.max_grad_norm > 0:
            # Handle DDP-wrapped policies
            raw = policy.module if hasattr(policy, "module") else policy
            if hasattr(raw, "trainable_parameters"):
                params = list(raw.trainable_parameters())
            else:
                params = [p for p in policy.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
        optimizer.step()
        self._step += 1

        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()}


@configclass
class SFTAlgorithmCfg(ModuleBaseCfg):
    """SFT algorithm configuration."""

    class_type: type[SFTAlgorithm] = SFTAlgorithm

    # Action loss
    action_loss_type: str = "mse"           # "mse", "l1", "smooth_l1"
    action_loss_weight: float = 1.0

    # KL regularization
    kl_coef: float = 0.0                    # 0 = no KL penalty

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
