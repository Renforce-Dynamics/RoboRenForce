from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Generator

from RoboRenForce import configclass
from dataclasses import MISSING

from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.components.actor import SACActor
from RoboRenForce.components.critic import MultiQNetwork
from RoboRenForce.networks.optimizer import GroupedOptimizerCfg, GroupedOptimizer
from RoboRenForce.algorithms.algorithm_base import AlgorithmBase, AlgorithmBaseCfg

class SAC(AlgorithmBase):
    """
    Soft Actor-Critic (off-policy).

    Assumes a direct transition replay buffer that yields dicts with keys:
        obs, critic_obs, actions, rewards, next_obs, next_critic_obs, termination, timeout

    - Asymmetric observations: policy_obs -> actor, critic_obs -> critics
    - Actor / critics / target critics built internally
    - Auto-tuned entropy temperature (optional)
    """
    cfg: "SACCfg"
    actor: SACActor
    critic: MultiQNetwork

    def __init__(
        self, cfg: SACCfg, actor: SACActor, critic: MultiQNetwork, device: str
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.actor = actor.to(device)
        self.critics = critic.to(device)
        self.target_critics = copy.deepcopy(self.critics).to(device)
        for p in self.target_critics.parameters():
            p.requires_grad_(False)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=cfg.critic_lr)
        self.optimizer = GroupedOptimizer(
            {
                "actor": self.actor_optimizer,
                "critic": self.critic_optimizer
            },
            self.cfg.optimzers_cfg
        )

        self.auto_entropy = cfg.auto_entropy
        if self.auto_entropy:
            self.target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(actor.action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=cfg.alpha_lr
            )
        else:
            self.alpha = torch.tensor(cfg.alpha, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, obs, critic_obs):
        return self.actor.sample(obs)[0]

    def update(self, generator: Generator[Dict[str, torch.Tensor], None, None]):
        """
        Consume mini-batches from a direct transition buffer and run SAC updates.

        Each yielded dict must contain:
            obs, critic_obs, actions, rewards,
            next_obs, next_critic_obs, termination, timeout
        """
        self.ptr_update = 0
        stats = {k: [] for k in [
            "critic_loss", "q_mean", "target_q_mean",
            "actor_loss", "alpha_loss", "alpha", "entropy",
        ]}

        for batch in generator:
            critic_loss, q_mean, target_q_mean, actor_loss, alpha_loss, alpha, entropy = \
                self._update_step(
                    obs             = batch["obs"],
                    critic_obs      = batch["critic_obs"],
                    actions         = batch["actions"],
                    rewards         = batch["rewards"],
                    next_obs        = batch["next_obs"],
                    next_critic_obs = batch["next_critic_obs"],
                    termination     = batch["termination"],
                    timeout         = batch["timeout"],
                )
            stats["critic_loss"].append(critic_loss)
            stats["q_mean"].append(q_mean)
            stats["target_q_mean"].append(target_q_mean)
            if actor_loss is not None:
                stats["actor_loss"].append(actor_loss)
            if alpha_loss is not None:
                stats["alpha_loss"].append(alpha_loss)
            if entropy is not None:
                stats["entropy"].append(entropy)
            stats["alpha"].append(alpha)
            self.ptr_update += 1

        def _mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "critic_loss":   _mean(stats["critic_loss"]),
            "actor_loss":    _mean(stats["actor_loss"]),
            "q_mean":        _mean(stats["q_mean"]),
            "target_q_mean": _mean(stats["target_q_mean"]),
            "alpha_loss":    _mean(stats["alpha_loss"]),
            "alpha":         _mean(stats["alpha"]),
            "entropy":       _mean(stats["entropy"]),
            "mini_batch_num": self.ptr_update,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_step(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        termination: torch.Tensor,
        timeout: torch.Tensor,
    ):
        critic_loss, q_mean, target_q_mean = self._update_critic(
            obs, critic_obs, actions, rewards,
            next_obs, next_critic_obs, termination, timeout,
        )
        if self._should_update_actor:
            actor_loss, alpha_loss, entropy = self._update_actor(obs, critic_obs)
        else:
            actor_loss, alpha_loss, entropy = None, None, None
        if self._should_update_target:
            self._soft_update()
        return critic_loss, q_mean, target_q_mean, actor_loss, alpha_loss, self.alpha.item(), entropy

    def _update_critic(
        self, obs, critic_obs, actions, rewards,
        next_obs, next_critic_obs, termination, timeout,
    ):
        gamma = self.cfg.gamma
        bootstrap_mask = 1.0 - termination.float()

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            target_q_min = torch.min(
                self.target_critics(next_critic_obs, next_actions), dim=0
            ).values
            target_q = rewards + gamma * bootstrap_mask * (
                target_q_min - self.alpha * next_log_prob
            )

        current_q_all = self.critics(critic_obs, actions)  # (num_q, B)
        critic_loss = sum((q - target_q).pow(2).mean() for q in current_q_all)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), self.cfg.max_grad_norm)
        self.critic_optimizer.step()

        return critic_loss.item(), current_q_all.mean().item(), target_q.mean().item()

    def _update_actor(self, obs, critic_obs):
        new_actions, log_prob = self.actor.sample(obs)
        q_new_min = torch.min(
            self.critics(critic_obs, new_actions), dim=0
        ).values
        actor_loss = (self.alpha.detach() * log_prob - q_new_min).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()

        alpha_loss = None
        if self.auto_entropy:
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        entropy = -log_prob.mean().item()
        return actor_loss.item(), (0.0 if alpha_loss is None else alpha_loss.item()), entropy

    @torch.no_grad()
    def _soft_update(self):
        tau = self.cfg.tau
        for p, tp in zip(self.critics.parameters(), self.target_critics.parameters()):
            tp.data.lerp_(p.data, tau)

    @property
    def _should_update_actor(self):
        return (self.ptr_update % self.cfg.actor_update_freq) == 0

    @property
    def _should_update_target(self):
        return (self.ptr_update % self.cfg.target_update_freq) == 0


@configclass
class SACCfg(AlgorithmBaseCfg):
    """SAC configuration."""

    class_type: type[SAC] = SAC
    optimzers_cfg: GroupedOptimizerCfg = GroupedOptimizerCfg()

    gamma               : float = 0.99
    tau                 : float = 0.005

    actor_lr            : float = 3e-4
    critic_lr           : float = 3e-4
    alpha_lr            : float = 3e-4
    actor_update_freq   : int = 4
    target_update_freq  : int = 4

    auto_entropy        : bool = True
    target_entropy      : float | None = None  # None -> -action_dim
    alpha               : float = 0.2

    max_grad_norm       : float = 1.0
