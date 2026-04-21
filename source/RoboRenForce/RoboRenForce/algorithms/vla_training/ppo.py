"""
PPO — Proximal Policy Optimization for VLA fine-tuning.

Actor-critic with Generalized Advantage Estimation (GAE).
Unlike GRPO (group-relative, no value function), PPO uses a learned value
function to compute per-step advantages via TD-error bootstrapping.

Algorithm:
    1. Collect rollouts with current policy, recording obs/actions/logprobs/values/rewards
    2. Compute GAE advantages: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
                                A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}
    3. Compute returns: R_t = A_t + V(s_t)
    4. For update_epochs:
         a. Re-evaluate logprobs, values under current policy
         b. Actor loss: clipped surrogate with GAE advantages
         c. Critic loss: clipped value loss (Huber)
         d. Optional: entropy bonus, KL penalty

Reference: RLinf rlinf/algorithms/losses.py, rlinf/algorithms/advantages.py
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: [T, B] per-step rewards
        values: [T+1, B] value estimates (includes bootstrap V(s_T))
        dones: [T, B] episode termination flags
        gamma: discount factor
        gae_lambda: GAE smoothing parameter
        normalize: whether to normalize advantages

    Returns:
        advantages: [T, B] GAE advantages
        returns: [T, B] target returns for value function
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], device=rewards.device)

    for t in reversed(range(T)):
        not_done = (~dones[t]).float()
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae

    returns = advantages + values[:-1]

    if normalize:
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

    return advantages, returns


def compute_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: Optional[float] = None,
    dual_clip_c: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO clipped surrogate actor loss.

    Args:
        logprobs: current policy log-probabilities
        old_logprobs: rollout log-probabilities
        advantages: GAE advantages
        clip_ratio_low: lower clip bound
        clip_ratio_high: upper clip bound (None = symmetric)
        dual_clip_c: dual clipping coefficient (None = disabled)

    Returns:
        loss: scalar
        metrics: dict with clip_fraction, approx_kl, ratio stats
    """
    clip_high = clip_ratio_high if clip_ratio_high is not None else clip_ratio_low
    log_ratio = logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_high)

    surr1 = -advantages * ratio
    surr2 = -advantages * clipped_ratio
    loss = torch.max(surr1, surr2)

    if dual_clip_c is not None:
        dual_clip_loss = dual_clip_c * advantages
        loss = torch.where(advantages < 0, torch.min(loss, dual_clip_loss), loss)

    loss = loss.mean()

    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio_low).float().mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()

    metrics = {
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
    }
    return loss, metrics


def compute_ppo_critic_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    value_clip: float = 0.2,
    huber_delta: float = 10.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO clipped value function loss with Huber.

    Args:
        values: current value predictions
        returns: target returns (advantages + old values)
        old_values: value predictions from rollout
        value_clip: clipping range for value function
        huber_delta: Huber loss delta

    Returns:
        loss: scalar
        metrics: dict with explained_variance
    """
    value_pred_clipped = old_values + torch.clamp(
        values - old_values, -value_clip, value_clip
    )

    loss_unclipped = F.huber_loss(values, returns, delta=huber_delta, reduction="none")
    loss_clipped = F.huber_loss(value_pred_clipped, returns, delta=huber_delta, reduction="none")
    loss = torch.max(loss_unclipped, loss_clipped).mean()

    with torch.no_grad():
        var_returns = returns.var()
        if var_returns > 1e-6:
            explained_var = 1.0 - (returns - values).var() / var_returns
        else:
            explained_var = torch.tensor(0.0)

    metrics = {"explained_variance": explained_var}
    return loss, metrics


class PPOAlgorithm(ModuleBase):
    """PPO algorithm for VLA RL fine-tuning with actor-critic.

    Usage:
        ppo = PPOAlgorithm(cfg)

        # After collecting rollouts with values:
        advantages, returns = ppo.compute_advantages(rewards, values, dones)
        metrics = ppo.update(logprobs, old_logprobs, advantages, values, returns, old_values, optimizer)
    """

    def __init__(self, cfg: "PPOAlgorithmCfg"):
        super().__init__()
        self.cfg = cfg
        self._step = 0

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        return compute_gae_advantages(
            rewards, values, dones,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            normalize=self.cfg.normalize_advantages,
        )

    def compute_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        entropy: Optional[torch.Tensor] = None,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute PPO actor-critic loss.

        Returns:
            dict with total_loss, policy_loss, value_loss, and metrics
        """
        # Actor loss
        policy_loss, actor_metrics = compute_ppo_actor_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_ratio_low=self.cfg.clip_ratio_low,
            clip_ratio_high=self.cfg.clip_ratio_high,
            dual_clip_c=self.cfg.dual_clip_c,
        )

        # Critic loss
        value_loss, critic_metrics = compute_ppo_critic_loss(
            values=values,
            returns=returns,
            old_values=old_values,
            value_clip=self.cfg.value_clip,
            huber_delta=self.cfg.huber_delta,
        )

        total_loss = policy_loss + self.cfg.value_loss_coef * value_loss

        result = {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            **{f"ppo/{k}": v for k, v in actor_metrics.items()},
            **{f"ppo/{k}": v for k, v in critic_metrics.items()},
        }

        # KL penalty
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
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        entropy: Optional[torch.Tensor] = None,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Compute loss, backprop, step optimizer."""
        loss_dict = self.compute_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            values=values,
            returns=returns,
            old_values=old_values,
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
class PPOAlgorithmCfg(ModuleBaseCfg):
    """PPO algorithm configuration."""

    class_type: type[PPOAlgorithm] = PPOAlgorithm

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True

    # PPO clipping
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    dual_clip_c: float = None
    value_clip: float = 0.2
    huber_delta: float = 10.0
    value_loss_coef: float = 0.5

    # KL / Entropy
    kl_beta: float = 0.0
    entropy_bonus: float = 0.01

    # Training
    update_epochs: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Reward
    reward_coef: float = 1.0
