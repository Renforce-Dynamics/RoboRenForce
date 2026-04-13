from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from RoboRenForce import configclass

from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.components.actor import StateIndStdActor
from RoboRenForce.components.critic import VNetwork
from RoboRenForce.buffer.online_rollout.rollout_storage import RolloutStorage
from RoboRenForce.utils.logging import timeit

from RoboRenForce.algorithms.algorithm_base import AlgorithmBase, AlgorithmBaseCfg

class PPO(AlgorithmBase):
    actor: StateIndStdActor
    critic: VNetwork

    def __init__(
        self,
        cfg: "PPOCfg",
        actor: "StateIndStdActor",
        critic: "VNetwork",
        device: str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # components
        self.actor = actor.to(device)
        self.critic = critic.to(device)

        # rollout storage (lazy init)
        self.storage = None


        # optimizer
        self.learning_rate = cfg.learning_rate
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate,
        )

    # --------------------------------------------------------------------- #
    # rollout & mode
    # --------------------------------------------------------------------- #
    
    def init_storage(
        self, *args, **kwargs
    ):
        def init_storage(
            num_envs: int,
            num_transitions_per_env: int,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
        ):
            self.storage = RolloutStorage(
                num_envs,
                num_transitions_per_env,
                actor_obs_shape,
                critic_obs_shape,
                action_shape,
                self.device,
            )
        storage: RolloutStorage = kwargs.get("storage", None)
        if storage is not None:
            self.storage: RolloutStorage = storage
        else:
            init_storage(*args, **kwargs)
        self.transition = self.storage.Transition()

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def test_mode(self):
        self.actor.eval()
        self.critic.eval()

    # --------------------------------------------------------------------- #
    # interaction
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def act(self, obs, critic_obs, **kwargs):
        self.transition.actions = self.actor.act(obs, **kwargs).detach()
        self.transition.values = self.critic(critic_obs, **kwargs).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # bootstrap on timeouts
        if "timeout" in infos:
            self.transition.rewards += (
                self.cfg.gamma
                * self.transition.values.squeeze(-1)
                * infos["timeout"].to(self.device)
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor.reset(dones); self.critic.reset(dones)

    @torch.no_grad()
    def compute_returns(self, last_critic_obs):
        last_values = self.critic(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    # --------------------------------------------------------------------- #
    # loss helpers
    # --------------------------------------------------------------------- #
    def _policy_forward(
        self,
        obs_batch,
        critic_obs_batch,
        actions_batch,
    ):
        """Forward pass through actor & critic for PPO update."""
        if isinstance(obs_batch, tuple):
            self.actor.act(*obs_batch)
        else:
            self.actor.act(obs_batch)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        entropy = self.actor.entropy()
        value_pred = self.critic(critic_obs_batch)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        return actions_log_prob_batch, entropy, value_pred, mu_batch, sigma_batch

    def _maybe_update_lr_with_kl(
        self,
        mu_batch,
        sigma_batch,
        old_mu_batch,
        old_sigma_batch,
    ):
        """Adaptive KL-based learning rate schedule.

        Matches the behavior of the original implementation:
        - decrease LR when KL is too large
        - increase LR when KL is too small
        using cfg.min_learning_rate / cfg.max_learning_rate as bounds.
        """
        if self.cfg.schedule != "adaptive" or self.cfg.desired_kl is None:
            return

        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                dim=-1,
            )
            kl_mean = torch.mean(kl)

            if kl_mean > self.cfg.desired_kl * 2.0:
                # KL too large -> decrease LR
                self.learning_rate = max(self.cfg.min_learning_rate, self.learning_rate / 1.5)
            elif kl_mean < self.cfg.desired_kl / 2.0 and kl_mean > 0.0:
                # KL too small but positive -> increase LR
                self.learning_rate = min(self.cfg.max_learning_rate, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _compute_surrogate_loss(
        self,
        actions_log_prob_batch,
        old_actions_log_prob_batch,
        advantages_batch,
    ):
        """Clipped PPO surrogate loss."""
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio,
            1.0 - self.cfg.clip_param,
            1.0 + self.cfg.clip_param,
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        return surrogate_loss

    def _compute_value_loss(
        self,
        value_pred,
        old_values_batch,
        returns_batch,
    ):
        """Value function loss, optionally clipped."""
        if self.cfg.use_clipped_value_loss:
            value_clipped = old_values_batch + (value_pred - old_values_batch).clamp(
                -self.cfg.clip_param, self.cfg.clip_param
            )
            value_loss = torch.max(
                (value_pred - returns_batch).pow(2),
                (value_clipped - returns_batch).pow(2),
            ).mean()
        else:
            value_loss = (returns_batch - value_pred).pow(2).mean()
        return value_loss

    def compute_loss(self, batch):
        """Compute PPO total loss and its components for a mini-batch."""
        (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            old_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) = batch

        # Forward pass
        actions_log_prob_batch, entropy, value_pred, mu_batch, sigma_batch = self._policy_forward(
            obs_batch=obs_batch,
            critic_obs_batch=critic_obs_batch,
            actions_batch=actions_batch,
        )

        # Adaptive KL & learning rate
        self._maybe_update_lr_with_kl(
            mu_batch=mu_batch,
            sigma_batch=sigma_batch,
            old_mu_batch=old_mu_batch,
            old_sigma_batch=old_sigma_batch,
        )

        # Loss terms
        surrogate_loss = self._compute_surrogate_loss(
            actions_log_prob_batch=actions_log_prob_batch,
            old_actions_log_prob_batch=old_actions_log_prob_batch,
            advantages_batch=advantages_batch,
        )
        value_loss = self._compute_value_loss(
            value_pred=value_pred,
            old_values_batch=old_values_batch,
            returns_batch=returns_batch,
        )
        entropy_loss = -self.cfg.entropy_coef * entropy.mean()

        total_loss = surrogate_loss + self.cfg.value_loss_coef * value_loss + entropy_loss

        return {
            "total_loss": total_loss,
            "surrogate_loss": surrogate_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
        }

    # --------------------------------------------------------------------- #
    # update
    # --------------------------------------------------------------------- #
    @timeit("update_time")
    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        generator = self.storage.mini_batch_generator(
            self.cfg.num_mini_batches, self.cfg.num_learning_epochs
        )
        for batch in generator:
            losses = self.compute_loss(batch)
            loss = losses["total_loss"]
            surrogate_loss = losses["surrogate_loss"]
            value_loss = losses["value_loss"]

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        self.storage.clear()
        
        return {
            "mean_value_loss" : mean_value_loss / num_updates,
            "mean_surrogate_loss" : mean_surrogate_loss / num_updates,
            "learning_rate": self.learning_rate,
            "mean_std": self.actor.action_std.mean()
        }

@configclass
class PPOCfg(AlgorithmBaseCfg):
    class_type: type[PPO] = PPO
    
    # optimization
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-2

    # PPO specific
    clip_param: float = 0.2
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0

    # return / advantage
    gamma: float = 0.99
    lam: float = 0.95
    use_clipped_value_loss: bool = True

    # KL control
    schedule: str = "fixed"  # fixed | adaptive
    desired_kl: float | None = 0.01