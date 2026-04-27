"""Algorithm presets tuned for RoboTwin manipulation."""

from __future__ import annotations

from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithmCfg
from RoboRenForce.algorithms.vla_training.ppo import PPOAlgorithmCfg


def grpo_robotwin_default() -> GRPOAlgorithmCfg:
    """GRPO defaults for RoboTwin (asymmetric clip, KL penalty)."""
    return GRPOAlgorithmCfg(
        group_size=8,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        kl_beta=0.05,
        update_epochs=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        reward_coef=1.0,
    )


def ppo_robotwin_default() -> PPOAlgorithmCfg:
    """PPO defaults for RoboTwin (value head + GAE)."""
    return PPOAlgorithmCfg(
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        value_loss_coef=0.5,
        learning_rate=1e-4,
        update_epochs=4,
        kl_beta=0.0,
    )
