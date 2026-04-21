"""
SAC — Soft Actor-Critic for off-policy VLA fine-tuning.

Maximum entropy RL: the policy maximizes expected return + entropy bonus.
Uses replay buffer for sample-efficient off-policy learning.

Algorithm:
    1. Collect experience with current policy → store in replay buffer
    2. Sample mini-batch from replay buffer
    3. Critic update: Q(s,a) → minimize MSE to target r + γ*(Q'(s',a') - α*log π(a'|s'))
    4. Actor update: π(a|s) → maximize Q(s,π(s)) - α*log π(a|s)
    5. Alpha update: minimize -α*(log π + target_entropy)
    6. Target Q: soft update with τ

Variants:
    - CrossQ: single forward pass for both Q(s,a) and Q(s',a')
    - RLPD: SAC + demonstration replay buffer
    - DSRL: dense supervised RL with separate encoders

Reference: RLinf rlinf/workers/actor/fsdp_sac_policy_worker.py
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


class SACQNetwork(nn.Module):
    """Double Q-network for SAC with configurable number of heads."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        num_heads: int = 2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            dims = [obs_dim + action_dim] + list(hidden_dims) + [1]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            self.heads.append(nn.Sequential(*layers))

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through all Q-heads.

        Args:
            obs: [B, obs_dim]
            actions: [B, action_dim]

        Returns:
            q_values: [B, num_heads]
        """
        x = torch.cat([obs, actions], dim=-1)
        qs = [head(x).squeeze(-1) for head in self.heads]
        return torch.stack(qs, dim=-1)


class SACGaussianPolicy(nn.Module):
    """Squashed Gaussian policy for SAC.

    Outputs tanh-squashed actions with proper log-probability correction.
    """

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

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log-probs with tanh squashing.

        Returns:
            actions: [B, action_dim] squashed actions
            log_probs: [B] log-probabilities
        """
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        # Reparameterized sample
        z = dist.rsample()
        actions = torch.tanh(z)

        # Log-prob with tanh correction
        log_probs = dist.log_prob(z).sum(dim=-1)
        log_probs -= (2 * (torch.log(torch.tensor(2.0)) - z - F.softplus(-2 * z))).sum(dim=-1)

        return actions, log_probs

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (mean, tanh-squashed)."""
        mean, _ = self(obs)
        return torch.tanh(mean)


class EntropyTemperature(nn.Module):
    """Learnable entropy temperature α for SAC.

    Supports three parameterizations:
    - "exp": α = exp(log_α)
    - "softplus": α = softplus(raw_α)
    - "fixed": constant α (no learning)
    """

    def __init__(
        self,
        initial_alpha: float = 0.1,
        alpha_type: str = "softplus",
    ):
        super().__init__()
        self.alpha_type = alpha_type

        if alpha_type == "exp":
            init_val = torch.log(torch.tensor(initial_alpha)).item()
            self.base_alpha = nn.Parameter(torch.tensor(init_val))
        elif alpha_type == "softplus":
            init_val = torch.log(torch.exp(torch.tensor(initial_alpha)) - 1).item()
            self.base_alpha = nn.Parameter(torch.tensor(init_val))
        elif alpha_type == "fixed":
            self.register_buffer("base_alpha", torch.tensor(initial_alpha))
        else:
            raise ValueError(f"Unknown alpha_type: {alpha_type}")

    @property
    def alpha(self) -> torch.Tensor:
        if self.alpha_type == "exp":
            return self.base_alpha.exp()
        elif self.alpha_type == "softplus":
            return F.softplus(self.base_alpha)
        else:
            return self.base_alpha


class ReplayBuffer:
    """Simple replay buffer for off-policy RL.

    Stores transitions (obs, action, reward, next_obs, done).
    """

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._obs: deque = deque(maxlen=max_size)
        self._actions: deque = deque(maxlen=max_size)
        self._rewards: deque = deque(maxlen=max_size)
        self._next_obs: deque = deque(maxlen=max_size)
        self._dones: deque = deque(maxlen=max_size)

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Add a batch of transitions.

        Args:
            obs: [B, obs_dim]
            actions: [B, action_dim]
            rewards: [B]
            next_obs: [B, obs_dim]
            dones: [B]
        """
        for i in range(obs.shape[0]):
            self._obs.append(obs[i].detach().cpu())
            self._actions.append(actions[i].detach().cpu())
            self._rewards.append(rewards[i].detach().cpu())
            self._next_obs.append(next_obs[i].detach().cpu())
            self._dones.append(dones[i].detach().cpu())

    def sample(self, batch_size: int, device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor]:
        """Sample a random batch.

        Returns:
            dict with obs, actions, rewards, next_obs, dones
        """
        total = len(self._obs)
        indices = torch.randint(0, total, (min(batch_size, total),))

        return {
            "obs": torch.stack([self._obs[i] for i in indices]).to(device),
            "actions": torch.stack([self._actions[i] for i in indices]).to(device),
            "rewards": torch.stack([self._rewards[i] for i in indices]).to(device),
            "next_obs": torch.stack([self._next_obs[i] for i in indices]).to(device),
            "dones": torch.stack([self._dones[i] for i in indices]).to(device),
        }

    def __len__(self):
        return len(self._obs)

    def is_ready(self, min_size: int = 100) -> bool:
        return len(self) >= min_size


class SACAlgorithm(ModuleBase):
    """Soft Actor-Critic algorithm for off-policy RL.

    Usage:
        sac = SACAlgorithm(cfg, obs_dim=16, action_dim=7)

        # Collect experience
        sac.add_experience(obs, actions, rewards, next_obs, dones)

        # Train
        if sac.is_ready():
            metrics = sac.update()
    """

    def __init__(self, cfg: "SACAlgorithmCfg", obs_dim: int, action_dim: int):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Networks
        self.actor = SACGaussianPolicy(
            obs_dim, action_dim, hidden_dims=cfg.hidden_dims,
        )
        self.critic = SACQNetwork(
            obs_dim, action_dim, hidden_dims=cfg.hidden_dims,
            num_heads=cfg.num_q_heads,
        )
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # Entropy temperature
        self.entropy_temp = EntropyTemperature(
            initial_alpha=cfg.initial_alpha,
            alpha_type=cfg.alpha_type,
        )
        self.target_entropy = -float(action_dim)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=cfg.replay_buffer_size)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr,
        )
        if cfg.alpha_type != "fixed":
            self.alpha_optimizer = torch.optim.Adam(
                [self.entropy_temp.base_alpha], lr=cfg.alpha_lr,
            )
        else:
            self.alpha_optimizer = None

        self._step = 0
        self._device = torch.device("cpu")

    def to(self, device):
        self._device = torch.device(device)
        return super().to(device)

    def add_experience(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Add experience to replay buffer."""
        self.replay_buffer.add(obs, actions, rewards, next_obs, dones)

    def is_ready(self) -> bool:
        return self.replay_buffer.is_ready(self.cfg.min_buffer_size)

    def _soft_update_target(self):
        tau = self.cfg.tau
        for target_p, online_p in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_p.data.mul_(1.0 - tau).add_(online_p.data, alpha=tau)

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Critic loss with entropy-augmented TD target.

        target_Q = r + γ * (1-d) * (min_i Q'_i(s', a') - α * log π(a'|s'))
        L_Q = Σ_i MSE(Q_i(s,a), target_Q)
        """
        alpha = self.entropy_temp.alpha.detach()

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            next_q = self.target_critic(next_obs, next_actions)

            # Aggregate Q-values
            if self.cfg.agg_q == "min":
                next_q_agg = next_q.min(dim=-1).values
            else:
                next_q_agg = next_q.mean(dim=-1)

            if self.cfg.backup_entropy:
                next_q_agg = next_q_agg - alpha * next_log_probs

            target_q = rewards + self.cfg.gamma * (~dones).float() * next_q_agg

        current_q = self.critic(obs, actions)  # [B, num_heads]
        critic_loss = F.mse_loss(
            current_q, target_q.unsqueeze(-1).expand_as(current_q)
        )

        metrics = {
            "q_mean": current_q.mean().detach(),
            "q_std": current_q.std().detach(),
            "target_q_mean": target_q.mean().detach(),
        }
        return critic_loss, metrics

    def compute_actor_loss(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Actor loss: maximize Q - α * log π.

        L_π = E[α * log π(a|s) - Q(s, a)]
        """
        alpha = self.entropy_temp.alpha.detach()

        actions, log_probs = self.actor.sample(obs)
        q_values = self.critic(obs, actions)

        if self.cfg.agg_q == "min":
            q_agg = q_values.min(dim=-1).values
        else:
            q_agg = q_values.mean(dim=-1)

        actor_loss = (alpha * log_probs - q_agg).mean()

        metrics = {
            "log_probs": log_probs.mean().detach(),
            "entropy": -log_probs.mean().detach(),
        }
        return actor_loss, metrics

    def compute_alpha_loss(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Entropy temperature loss.

        L_α = -α * (log π(a|s) + target_entropy)
        """
        alpha = self.entropy_temp.alpha

        with torch.no_grad():
            _, log_probs = self.actor.sample(obs)

        alpha_loss = -(alpha * (log_probs.mean() + self.target_entropy))

        metrics = {"alpha": alpha.detach()}
        return alpha_loss, metrics

    def update(self) -> dict[str, float]:
        """Full SAC update step.

        Order: critic → actor → alpha → target

        Returns:
            dict with scalar metrics
        """
        batch = self.replay_buffer.sample(self.cfg.batch_size, self._device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # 1. Critic update
        critic_loss, c_metrics = self.compute_critic_loss(
            obs, actions, rewards, next_obs, dones
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Actor update
        actor_loss, a_metrics = self.compute_actor_loss(obs)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Alpha update
        alpha_metrics = {}
        if self.alpha_optimizer is not None:
            alpha_loss, alpha_metrics = self.compute_alpha_loss(obs)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # 4. Target update
        if self._step % self.cfg.target_update_freq == 0:
            self._soft_update_target()

        self._step += 1

        metrics = {
            "total_loss": (critic_loss + actor_loss).item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            **{f"sac/{k}": v.item() if isinstance(v, torch.Tensor) else v
               for d in [c_metrics, a_metrics, alpha_metrics] for k, v in d.items()},
        }
        return metrics

    @torch.no_grad()
    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select action from current policy.

        Args:
            obs: [B, obs_dim]
            deterministic: if True, use mean action

        Returns:
            actions: [B, action_dim]
        """
        if deterministic:
            return self.actor.deterministic(obs)
        actions, _ = self.actor.sample(obs)
        return actions


@configclass
class SACAlgorithmCfg(ModuleBaseCfg):
    """SAC algorithm configuration."""

    class_type: type[SACAlgorithm] = SACAlgorithm

    # Network architecture
    hidden_dims: tuple[int, ...] = (256, 256)
    num_q_heads: int = 2
    agg_q: str = "min"                  # "min" or "mean"

    # SAC hyperparameters
    gamma: float = 0.99
    tau: float = 0.005
    target_update_freq: int = 1
    backup_entropy: bool = True

    # Entropy
    initial_alpha: float = 0.1
    alpha_type: str = "softplus"        # "softplus", "exp", "fixed"

    # Replay buffer
    replay_buffer_size: int = 100000
    min_buffer_size: int = 100
    batch_size: int = 256

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
