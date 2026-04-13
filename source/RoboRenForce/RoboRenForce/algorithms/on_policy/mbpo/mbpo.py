from __future__ import annotations
from dataclasses import MISSING

import torch
import torch.nn as nn
import torch.optim as optim

from RoboRenForce import configclass

from RoboRenForce.components.actor import StateIndStdActor
from RoboRenForce.components.critic import VNetwork
from RoboRenForce.components.actor_critic_pack import ActorCritic
from RoboRenForce.components.normalizer import NormalizerBase
from RoboRenForce.utils.template.module_base import ModuleBase
from RoboRenForce.buffer.online_rollout.rollout_storage import RolloutStorage

from RoboRenForce.algorithms.on_policy.ppo import PPO, PPOCfg

class MBPO(PPO):
    """Model-based PPO (MBPO) on-policy algorithm.

    Extends PPO with a learned system dynamics model and optional imagination rollouts.
    """

    actor: StateIndStdActor
    critic: VNetwork

    def __init__(
        self,
        cfg                 : "MBPOCfg",
        actor               : "StateIndStdActor",
        critic              : "VNetwork",
        device              : str,
    ):
        super().__init__(cfg=cfg, actor=actor, critic=critic, device=device)

        self.cfg = cfg
        self.imagination_storage: RolloutStorage | None = None
        self.imagination_transition = RolloutStorage.Transition()

    def init_imagination_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        """Initialize rollout storage for imagination trajectories."""
        self.imagination_storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    # --------------------------------------------------------------------- #
    # interaction with env / imagination
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def act_imagination(self, obs, critic_obs):
        """Act in imagination; mirrors `act` but writes into imagination transition."""
        self.imagination_transition.actions = self.actor.act(obs).detach()
        self.imagination_transition.values = self.critic(critic_obs).detach()
        self.imagination_transition.actions_log_prob = (
            self.actor.get_actions_log_prob(self.imagination_transition.actions).detach()
        )
        self.imagination_transition.action_mean = self.actor.action_mean.detach()
        self.imagination_transition.action_sigma = self.actor.action_std.detach()
        self.imagination_transition.observations = obs
        self.imagination_transition.critic_observations = critic_obs
        return self.imagination_transition.actions

    def process_env_step(
        self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict, imagination: bool = False,
    ):
        if imagination:
            # Mirror PPO.process_env_step but using imagination_transition/storage.
            self.imagination_transition.rewards = rewards.clone()
            self.imagination_transition.dones = dones
            if "timeout" in infos:
                self.imagination_transition.rewards += (
                    self.cfg.gamma
                    * self.imagination_transition.values.squeeze(-1)
                    * infos["timeout"].to(self.device)
                )
            self.imagination_storage.add_transitions(self.imagination_transition)
            self.imagination_transition.clear()
            self.actor.reset(dones)
            self.critic.reset(dones)
        else:
            return super().process_env_step(rewards, dones, infos)

    @torch.no_grad()
    def compute_imagination_returns(self, last_critic_obs: torch.Tensor):
        """Compute returns for imagination trajectories."""
        last_values = self.critic(last_critic_obs).detach()
        self.imagination_storage.compute_returns(
            last_values, self.cfg.gamma, self.cfg.lam
        )

    # --------------------------------------------------------------------- #
    # PPO update with optional imagination data
    # --------------------------------------------------------------------- #

    def _combined_mini_batch_generator(self):
        """Combine real and imagination storage mini-batches by concatenation."""
        assert self.imagination_storage is not None

        real_gen = self.storage.mini_batch_generator(
            self.cfg.num_mini_batches, self.cfg.num_learning_epochs
        )
        imag_gen = self.imagination_storage.mini_batch_generator(
            self.cfg.num_mini_batches, self.cfg.num_learning_epochs
        )

        for real_batch, imag_batch in zip(real_gen, imag_gen):
            (
                obs_r,
                critic_obs_r,
                actions_r,
                old_values_r,
                advantages_r,
                returns_r,
                old_log_prob_r,
                old_mu_r,
                old_sigma_r,
            ) = real_batch
            (
                obs_i,
                critic_obs_i,
                actions_i,
                old_values_i,
                advantages_i,
                returns_i,
                old_log_prob_i,
                old_mu_i,
                old_sigma_i,
            ) = imag_batch

            obs = torch.cat([obs_r, obs_i], dim=0)
            critic_obs = torch.cat([critic_obs_r, critic_obs_i], dim=0)
            actions = torch.cat([actions_r, actions_i], dim=0)
            old_values = torch.cat([old_values_r, old_values_i], dim=0)
            advantages = torch.cat([advantages_r, advantages_i], dim=0)
            returns = torch.cat([returns_r, returns_i], dim=0)
            old_log_prob = torch.cat([old_log_prob_r, old_log_prob_i], dim=0)
            old_mu = torch.cat([old_mu_r, old_mu_i], dim=0)
            old_sigma = torch.cat([old_sigma_r, old_sigma_i], dim=0)

            yield (
                obs,
                critic_obs,
                actions,
                old_values,
                advantages,
                returns,
                old_log_prob,
                old_mu,
                old_sigma,
            )

    def update(self, use_imagination: bool = False):
        """Run a PPO update; optionally mix in imagination data."""
        if not use_imagination or self.imagination_storage is None:
            return super().update()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0

        generator = self._combined_mini_batch_generator()

        for batch in generator:
            # Reuse PPO's decoupled loss computation on the combined real+imagined batch.
            losses = self.compute_loss(batch)
            loss = losses["total_loss"]
            surrogate_loss = losses["surrogate_loss"]
            value_loss = losses["value_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()

            mean_value_loss += float(value_loss.detach())
            mean_surrogate_loss += float(surrogate_loss.detach())

        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        self.storage.clear()
        if self.imagination_storage is not None:
            self.imagination_storage.clear()

        return {
            "mean_value_loss": mean_value_loss / num_updates,
            "mean_surrogate_loss": mean_surrogate_loss / num_updates,
            "learning_rate": self.learning_rate,
            "mean_std": self.actor.action_std.mean(),
        }


@configclass
class MBPOCfg(PPOCfg):
    class_type: type["MBPO"] = MBPO  # set in __post_init__
    
    def construct_from_cfg(
        self,
        actor_critic: "ActorCritic",
        device: str, *args, **kwargs,
    ):
        return self.class_type(
            cfg=self,
            actor=actor_critic.actor,
            critic=actor_critic.critic,
            device=device, *args, **kwargs,
        )
