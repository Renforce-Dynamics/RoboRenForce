"""
IQL — Implicit Q-Learning for offline VLA fine-tuning.

Offline RL algorithm that avoids querying out-of-distribution actions
by using expectile regression for the value function and advantage-weighted
regression for the actor.

Algorithm:
    1. Critic: Q(s,a) → minimize MSE to target r + γ * V(s')
    2. Value: V(s) → minimize expectile loss L_τ(min(Q1,Q2) - V(s))
    3. Actor: π(a|s) → maximize exp(β*(Q-V)) * log π(a|s)
    4. Target Q: soft update with τ

Key insight: the expectile loss with τ > 0.5 approximates max-Q
without requiring maximization over actions.

Reference: RLinf rlinf/workers/actor/fsdp_iql_policy_worker.py
"""

from __future__ import annotations

from typing import Optional
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


def iql_expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Expectile regression loss.

    L = |τ - I(δ < 0)| * δ²

    When τ > 0.5, this upweights positive errors (overestimates),
    pushing V(s) toward upper quantiles of Q(s,a).

    Args:
        diff: Q_target - V(s)
        expectile: τ parameter (typically 0.7-0.9)

    Returns:
        per-element expectile loss
    """
    neg_mask = (diff < 0).float()
    return torch.abs(expectile - neg_mask) * diff.square()


class QNetwork(nn.Module):
    """Double Q-network for IQL."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        dims = [obs_dim + action_dim] + list(hidden_dims) + [1]
        layers_q1 = []
        layers_q2 = []
        for i in range(len(dims) - 1):
            layers_q1.append(nn.Linear(dims[i], dims[i + 1]))
            layers_q2.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers_q1.append(nn.ReLU())
                layers_q2.append(nn.ReLU())
        self.q1 = nn.Sequential(*layers_q1)
        self.q2 = nn.Sequential(*layers_q2)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, actions], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ValueNetwork(nn.Module):
    """Value network V(s) for IQL."""

    def __init__(self, obs_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        dims = [obs_dim] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class GaussianPolicy(nn.Module):
    """Gaussian policy for IQL actor."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        dims = [obs_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(dims[-1], action_dim)
        self.log_std_head = nn.Linear(dims[-1], action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1)

    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.rsample()


class IQLAlgorithm(ModuleBase):
    """Implicit Q-Learning algorithm for offline RL.

    Usage:
        iql = IQLAlgorithm(cfg, obs_dim=16, action_dim=7)
        metrics = iql.update(batch)
    """

    def __init__(self, cfg: "IQLAlgorithmCfg", obs_dim: int, action_dim: int):
        super().__init__()
        self.cfg = cfg

        # Networks
        self.actor = GaussianPolicy(
            obs_dim, action_dim,
            hidden_dims=cfg.hidden_dims,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        )
        self.critic = QNetwork(obs_dim, action_dim, hidden_dims=cfg.hidden_dims)
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.value = ValueNetwork(obs_dim, hidden_dims=cfg.hidden_dims)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=cfg.value_lr
        )

        self._step = 0

    def _soft_update_target(self):
        """Exponential moving average update for target critic."""
        tau = self.cfg.tau
        for target_p, online_p in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_p.data.mul_(1.0 - tau).add_(online_p.data, alpha=tau)

    def compute_value_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Value function loss with expectile regression.

        L_V = E[L_τ(min(Q1_target, Q2_target) - V(s))]
        """
        with torch.no_grad():
            q1_t, q2_t = self.target_critic(obs, actions)
            q_target = torch.min(q1_t, q2_t)

        v = self.value(obs)
        value_loss = iql_expectile_loss(q_target - v, self.cfg.expectile).mean()

        return value_loss, {"value": v.mean().detach()}

    def compute_actor_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Actor loss with advantage-weighted regression.

        L_π = -E[exp(β * (Q - V)) * log π(a|s)]
        """
        with torch.no_grad():
            v = self.value(obs)
            q1_t, q2_t = self.target_critic(obs, actions)
            q_target = torch.min(q1_t, q2_t)
            adv = q_target - v
            exp_adv = torch.exp(self.cfg.temperature * adv).clamp(max=100.0)

        log_probs = self.actor.log_prob(obs, actions)
        actor_loss = -(exp_adv * log_probs).mean()

        return actor_loss, {"adv_mean": adv.mean().detach()}

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Critic loss with TD target from value function.

        L_Q = E[(Q(s,a) - (r + γ * (1-d) * V(s')))²]
        """
        with torch.no_grad():
            next_v = self.value(next_obs)
            target_q = rewards + self.cfg.gamma * (~dones).float() * next_v

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        return critic_loss, {
            "q1_mean": q1.mean().detach(),
            "q2_mean": q2.mean().detach(),
        }

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Full IQL update step.

        Update order: value → actor → critic → target

        Args:
            batch: dict with obs, actions, rewards, next_obs, dones

        Returns:
            dict with scalar metrics
        """
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # 1. Value update
        value_loss, v_metrics = self.compute_value_loss(obs, actions)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 2. Actor update
        actor_loss, a_metrics = self.compute_actor_loss(obs, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Critic update
        critic_loss, c_metrics = self.compute_critic_loss(
            obs, actions, rewards, next_obs, dones
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 4. Target update
        self._soft_update_target()
        self._step += 1

        metrics = {
            "total_loss": (value_loss + actor_loss + critic_loss).item(),
            "value_loss": value_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            **{f"iql/{k}": v.item() if isinstance(v, torch.Tensor) else v
               for d in [v_metrics, a_metrics, c_metrics] for k, v in d.items()},
        }
        return metrics

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select action from current policy.

        Args:
            obs: [B, obs_dim]
            deterministic: if True, use mean (no sampling)

        Returns:
            actions: [B, action_dim]
        """
        if deterministic:
            mean, _ = self.actor(obs)
            return mean
        return self.actor.sample(obs)


@configclass
class IQLAlgorithmCfg(ModuleBaseCfg):
    """IQL algorithm configuration."""

    class_type: type[IQLAlgorithm] = IQLAlgorithm

    # Network architecture
    hidden_dims: tuple[int, ...] = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    # IQL hyperparameters
    expectile: float = 0.7              # τ for value function (0.5-0.9)
    temperature: float = 3.0            # β for advantage weighting
    gamma: float = 0.99                 # discount factor
    tau: float = 0.005                  # soft update coefficient

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
