from __future__ import annotations

from dataclasses import MISSING
from typing import Dict, Tuple

import torch
import torch.nn as nn

from RoboRenForce import configclass
from RoboRenForce.algorithms.on_policy.ppo import PPO, PPOCfg
from RoboRenForce.components.actor.state_ind_std_actor import StateIndStdActor
from RoboRenForce.components.critic import VNetwork
from RoboRenForce.components.discriminator import Discriminator, DiscriminatorCfg
from RoboRenForce.utils.logging import timeit
from RoboRenForce.utils.normalizer import RunningMeanStd
from RoboRenForce.buffer.direct_based.transition_buffer import AMPTransitionBuffer


class AMPPPO(PPO):
    """AMP-style PPO built on top of the standard PPO implementation.

    This class reuses PPO's rollout storage, action selection and value/advantage
    computation, and only extends:
        - additional AMP transition storage (s_t, s_{t+1})
        - discriminator loss terms in the PPO update.
    """

    discriminator: Discriminator

    def __init__(
        self,
        cfg: "AMPPPOCfg",
        actor: StateIndStdActor,
        critic: VNetwork,
        discriminator: Discriminator,
        amp_normalizer: RunningMeanStd | None,
        device: str = "cpu",
    ):
        # Initialize standard PPO components
        super().__init__(cfg=cfg, actor=actor, critic=critic, device=device)

        # AMP-specific components
        self.discriminator: Discriminator = discriminator.to(device)
        self.amp_normalizer = amp_normalizer
        self.amp_storage = AMPTransitionBuffer(
            obs_dim=cfg.amp_obs_dim,
            capacity=cfg.amp_replay_buffer_size,
            device=device,
        )

        # Extend optimizer to include discriminator parameters
        params = [
            {"params": self.actor.parameters(), "name": "actor"},
            {"params": self.critic.parameters(), "name": "critic"},
            {
                "params": self.discriminator.trunk.parameters(),
                "weight_decay": 1e-4,
                "name": "amp_trunk",
            },
            {
                "params": self.discriminator.amp_linear.parameters(),
                "weight_decay": 1e-2,
                "name": "amp_head",
            },
        ]
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        # AMP transition container (reuses RolloutStorage.Transition API)
        from RoboRenForce.buffer.online_rollout.rollout_storage import RolloutStorage

        self.amp_transition = RolloutStorage.Transition()

    # ------------------------------------------------------------------ #
    # Interaction (extend PPO.act / process_env_step)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs, critic_obs, amp_obs):
        """Select actions via PPO and additionally record AMP observations."""
        actions = super().act(obs, critic_obs)
        # Record AMP observations corresponding to the current state
        self.amp_transition.observations = amp_obs
        return actions

    def process_env_step(self, rewards, dones, infos, amp_next_obs):
        """Process environment step and push AMP transitions to AMP buffer.

        The PPO part (rewards, dones, storage) is handled by the parent class.
        """
        # Insert AMP transition (s_t, s_{t+1}) before PPO clears transitions
        self.amp_storage.insert(self.amp_transition.observations, amp_next_obs)

        # Delegate reward processing and PPO storage handling to base PPO
        super().process_env_step(rewards, dones, infos)

        # Clear AMP transition after it has been used
        self.amp_transition.clear()

    # ------------------------------------------------------------------ #
    # AMP reward shaping (beyondAMP-style)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_amp_reward(
        self,
        amp_state: torch.Tensor,
        amp_next_state: torch.Tensor,
        task_rewards: torch.Tensor,
    ):
        """Compute AMP-style shaped reward following beyondAMP formulation.

        AMP reward:
            r_amp = amp_reward_coef * clamp(1 - 0.25 * (D(s, s') - 1)^2, min=0)
        Final reward:
            r = lerp(r_amp, task_reward; amp_task_reward_lerp)
        """
        # Optional normalization
        if self.amp_normalizer is not None:
            amp_state = self.amp_normalizer.normalize(amp_state)
            amp_next_state = self.amp_normalizer.normalize(amp_next_state)

        disc_input = torch.cat([amp_state, amp_next_state], dim=-1)
        d_logits = self.discriminator(disc_input)

        # BeyondAMP quadratic reward around D=1
        amp_rewards = self.cfg.amp_reward_coef * torch.clamp(
            1.0 - 0.25 * torch.square(d_logits - 1.0), min=0.0
        )

        # Discriminator output is (num_envs, 1); task_rewards comes from the
        # env as (num_envs,). Promote task_rewards to (num_envs, 1) so the
        # lerp is element-wise instead of broadcasting (num_envs,) +
        # (num_envs, 1) into a (num_envs, num_envs) matrix.
        if task_rewards.dim() == 1:
            task_r = task_rewards.unsqueeze(-1)
        else:
            task_r = task_rewards

        reward = amp_rewards
        if self.cfg.amp_task_reward_lerp > 0.0:
            reward = (
                (1.0 - self.cfg.amp_task_reward_lerp) * amp_rewards
                + self.cfg.amp_task_reward_lerp * task_r
            )

        # For logging, we return per-env scalars
        return reward.squeeze(-1), d_logits, amp_rewards.squeeze(-1)

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    @timeit("update_time")
    def update(self) -> Dict[str, float]:
        """Perform one AMP PPO update over the collected rollout.

        This method mirrors the PPO update but augments it with:
            - discriminator loss (policy vs expert transitions)
            - gradient penalty regularization.
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0

        generator = self.storage.mini_batch_generator(
            self.cfg.num_mini_batches, self.cfg.num_learning_epochs
        )

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.cfg.num_learning_epochs * self.cfg.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env
            // self.cfg.num_mini_batches,
        )

        amp_expert_generator = self.cfg.motion_dataset.feed_forward_generator(
            self.cfg.num_learning_epochs * self.cfg.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env
            // self.cfg.num_mini_batches,
        )

        for batch, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):
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

            # PPO policy forward pass (reuse PPO actor/critic)
            self.actor.act(obs_batch)
            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            entropy = self.actor.entropy()
            value_pred = self.critic(critic_obs_batch)
            mu_batch = self.actor.action_mean
            sigma_batch = self.actor.action_std

            # Adaptive KL schedule (same as PPO)
            if self.cfg.schedule == "adaptive" and self.cfg.desired_kl is not None:
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + 1e-8) + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(self.cfg.min_learning_rate, self.learning_rate / 1.5)
                    elif kl_mean < self.cfg.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(self.cfg.max_learning_rate, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # PPO surrogate loss (identical to PPO)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.cfg.clip_param,
                1.0 + self.cfg.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss (identical to PPO)
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

            # Discriminator loss (policy vs expert transitions)
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize(policy_state)
                    policy_next_state = self.amp_normalizer.normalize(policy_next_state)
                    expert_state = self.amp_normalizer.normalize(expert_state)
                    expert_next_state = self.amp_normalizer.normalize(expert_next_state)

            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

            expert_loss = nn.functional.mse_loss(
                expert_d, torch.ones_like(expert_d, device=self.device)
            )
            policy_loss = nn.functional.mse_loss(
                policy_d, -1 * torch.ones_like(policy_d, device=self.device)
            )
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state, expert_next_state, lambda_=10.0
            )

            # Total loss: PPO objective + AMP discriminator regularization
            loss = (
                surrogate_loss
                + self.cfg.value_loss_coef * value_loss
                - self.cfg.entropy_coef * entropy.mean()
                + amp_loss
                + grad_pen_loss
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.detach())
                self.amp_normalizer.update(expert_state.detach())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        self.storage.clear()

        if num_updates == 0:
            return {}

        return {
            "mean_value_loss": mean_value_loss / num_updates,
            "mean_surrogate_loss": mean_surrogate_loss / num_updates,
            "mean_amp_loss": mean_amp_loss / num_updates,
            "mean_grad_pen_loss": mean_grad_pen_loss / num_updates,
            "mean_policy_pred": mean_policy_pred / num_updates,
            "mean_expert_pred": mean_expert_pred / num_updates,
            "learning_rate": self.learning_rate,
            "mean_std": self.actor.action_std.mean().item(),
        }


@configclass
class AMPPPOCfg(PPOCfg):
    """Configuration for AMPPPO algorithm."""

    class_type: type[AMPPPO] = AMPPPO

    # AMP specific (in addition to standard PPO settings)
    amp_replay_buffer_size: int = 100000
    amp_reward_coef: float = 1.0
    amp_discr_hidden_dims: Tuple[int, ...] = (256, 256)
    amp_task_reward_lerp: float = 0.5

    # Runtime-provided (not set in static config)
    amp_obs_dim: int = MISSING
    motion_dataset: object = MISSING

    def construct_from_cfg(
        self,
        actor_critic,
        discriminator,
        amp_normalizer: RunningMeanStd | None,
        motion_dataset,
        amp_obs_dim: int,
        device,
        *args,
        **kwargs,
    ):
        """Custom factory for AMPPPO with additional dependencies."""
        self.motion_dataset = motion_dataset
        self.amp_obs_dim = amp_obs_dim
        return self.class_type(
            self,
            actor=actor_critic.actor,
            critic=actor_critic.critic,
            discriminator=discriminator,
            amp_normalizer=amp_normalizer,
            device=device,
        )


__all__ = ["AMPPPO", "AMPPPOCfg"]

