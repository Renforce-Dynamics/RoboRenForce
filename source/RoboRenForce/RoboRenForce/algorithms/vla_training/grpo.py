"""
GRPO — Group Relative Policy Optimization for VLA fine-tuning.

Core idea: collect multiple rollouts per task (a "group"), normalize rewards
within each group to get advantages, then apply PPO-style clipped surrogate
loss. No value function needed.

Reference: RLinf rlinf/algorithms/advantages.py, rlinf/algorithms/losses.py

Algorithm:
    1. For each task, run group_size rollouts with the current policy
    2. Compute group-relative advantages:
       adv_i = (reward_i - mean(group_rewards)) / (std(group_rewards) + eps)
    3. Re-evaluate logprobs under current policy
    4. Compute clipped surrogate loss (same as PPO):
       ratio = exp(log_pi - log_pi_old)
       loss = -min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)
    5. Optional: KL penalty against reference policy
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Compute group-relative advantages.

    Args:
        rewards: [num_groups * group_size] or [num_groups, group_size]
        group_size: number of rollouts per group

    Returns:
        advantages: same shape as rewards, z-normalized within each group
    """
    original_shape = rewards.shape
    grouped = rewards.view(-1, group_size)

    group_mean = grouped.mean(dim=-1, keepdim=True)
    group_std = grouped.std(dim=-1, keepdim=True)

    advantages = (grouped - group_mean) / (group_std + 1e-6)
    return advantages.view(original_shape)


def compute_clipped_surrogate_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
    clip_ratio_high: Optional[float] = None,
    dual_clip_c: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO-style clipped surrogate loss.

    Args:
        logprobs: [B] current policy log-probabilities
        old_logprobs: [B] old policy log-probabilities (from rollout)
        advantages: [B] advantages (group-relative for GRPO)
        clip_ratio: clipping epsilon (symmetric if clip_ratio_high is None)
        clip_ratio_high: upper clip bound (asymmetric clipping)
        dual_clip_c: dual clipping coefficient (optional)

    Returns:
        loss: scalar
        metrics: dict with clip_fraction, approx_kl, ratio stats
    """
    log_ratio = logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    clip_high = clip_ratio_high if clip_ratio_high is not None else clip_ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_high)

    surr1 = -advantages * ratio
    surr2 = -advantages * clipped_ratio
    loss = torch.max(surr1, surr2)

    # Optional dual clipping (for negative advantages)
    if dual_clip_c is not None:
        dual_clip_loss = dual_clip_c * advantages
        loss = torch.where(advantages < 0, torch.min(loss, dual_clip_loss), loss)

    loss = loss.mean()

    # Metrics
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()

    metrics = {
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
    }

    return loss, metrics


class GRPOAlgorithm(ModuleBase):
    """GRPO algorithm for VLA RL fine-tuning.

    Usage:
        grpo = GRPOAlgorithm(cfg)

        # After collecting rollouts:
        advantages = grpo.compute_advantages(rewards)
        loss_dict = grpo.update(batch, policy, optimizer)
    """

    def __init__(self, cfg: "GRPOAlgorithmCfg"):
        super().__init__()
        self.cfg = cfg
        self._step = 0

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages from episode rewards."""
        return compute_grpo_advantages(rewards, self.cfg.group_size)

    def compute_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        entropy: Optional[torch.Tensor] = None,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute GRPO loss.

        Args:
            logprobs: [B] current policy log-probabilities
            old_logprobs: [B] rollout log-probabilities
            advantages: [B] group-relative advantages
            entropy: [B] policy entropy (optional, for bonus)
            ref_logprobs: [B] reference policy log-probs (optional, for KL penalty)

        Returns:
            dict with loss, policy_loss, and optional kl_loss, entropy_bonus
        """
        policy_loss, metrics = compute_clipped_surrogate_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_ratio=self.cfg.clip_ratio_low,
            clip_ratio_high=self.cfg.clip_ratio_high,
            dual_clip_c=self.cfg.dual_clip_c,
        )

        total_loss = policy_loss

        result = {
            "policy_loss": policy_loss.detach(),
            **{f"grpo/{k}": v for k, v in metrics.items()},
        }

        # KL penalty against reference policy
        if ref_logprobs is not None and self.cfg.kl_beta > 0:
            kl_div = (old_logprobs - ref_logprobs).mean()
            kl_loss = self.cfg.kl_beta * kl_div
            total_loss = total_loss + kl_loss
            result["kl_loss"] = kl_loss.detach()
            result["kl_div"] = kl_div.detach()

        # Entropy bonus
        if entropy is not None and self.cfg.entropy_bonus > 0:
            entropy_loss = -self.cfg.entropy_bonus * entropy.mean()
            total_loss = total_loss + entropy_loss
            result["entropy"] = entropy.mean().detach()
            result["entropy_loss"] = entropy_loss.detach()

        result["total_loss"] = total_loss
        return result

    def update(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        entropy: Optional[torch.Tensor] = None,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Compute loss, backprop, step optimizer.

        Returns:
            dict with scalar metrics for logging
        """
        loss_dict = self.compute_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            entropy=entropy,
            ref_logprobs=ref_logprobs,
        )

        total_loss = loss_dict["total_loss"]

        optimizer.zero_grad()
        total_loss.backward()
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                self.cfg.max_grad_norm,
            )
        optimizer.step()

        self._step += 1

        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()}


@configclass
class GRPOAlgorithmCfg(ModuleBaseCfg):
    """GRPO algorithm configuration."""

    class_type: type[GRPOAlgorithm] = GRPOAlgorithm

    # Group
    group_size: int = 8             # rollouts per task for advantage normalization

    # PPO clipping
    clip_ratio_low: float = 0.2     # lower clip bound
    clip_ratio_high: float = 0.28   # upper clip bound (asymmetric)
    dual_clip_c: float = None       # dual clipping coeff (None = disabled)

    # KL penalty
    kl_beta: float = 0.05           # KL penalty coefficient
    kl_type: str = "kl"             # "kl" or "abs" or "mse"

    # Entropy
    entropy_bonus: float = 0.05     # entropy bonus coefficient

    # Training
    update_epochs: int = 2          # PPO epochs per rollout batch
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Reward
    reward_coef: float = 5.0        # scale raw rewards
    gamma: float = 1.0              # discount factor (1.0 = no discounting)
