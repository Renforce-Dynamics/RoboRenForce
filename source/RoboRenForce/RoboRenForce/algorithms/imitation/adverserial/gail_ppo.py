from __future__ import annotations

from dataclasses import MISSING
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from RoboRenForce import configclass
from RoboRenForce.algorithms.on_policy.ppo import PPO, PPOCfg
from RoboRenForce.components.actor.state_ind_std_actor import StateIndStdActor
from RoboRenForce.components.critic import VNetwork
from RoboRenForce.components.discriminator import Discriminator, DiscriminatorCfg
from RoboRenForce.utils.logging import timeit
from RoboRenForce.utils.normalizer import RunningMeanStd
from RoboRenForce.buffer.direct_based.transition_buffer import AMPTransitionBuffer


class GAILPPO(PPO):
    """GAIL + PPO algorithm implemented on top of the core PPO class.

    This class reuses PPO's rollout storage and PPO loss computation, and extends it with:
        - discriminator updates on expert vs policy AMP transitions
        - discriminator-based GAIL rewards added to returns and advantages.
    """

    discriminator: Discriminator

    def __init__(
        self,
        cfg: "GAILPPOCfg",
        actor: StateIndStdActor,
        critic: VNetwork,
        discriminator: Discriminator,
        amp_normalizer: RunningMeanStd | None,
        device: str = "cpu",
    ):
        super().__init__(cfg=cfg, actor=actor, critic=critic, device=device)

        self.discriminator: Discriminator = discriminator.to(device)
        self.amp_normalizer = amp_normalizer

        from RoboRenForce.buffer.online_rollout.rollout_storage import RolloutStorage

        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = AMPTransitionBuffer(
            obs_dim=cfg.amp_obs_dim,
            capacity=cfg.amp_replay_buffer_size,
            device=device,
        )

        # Separate optimizers for policy (PPO) and discriminator
        self.policy_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.policy_lr,
        )
        disc_params = list(self.discriminator.trunk.parameters()) + list(
            self.discriminator.amp_linear.parameters()
        )
        self.disc_optimizer = torch.optim.Adam(disc_params, lr=cfg.disc_lr)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.min_std = cfg.min_std

    @torch.no_grad()
    def act(self, obs, critic_obs, amp_obs):
        """Select actions via PPO and additionally record AMP observations."""
        actions = super().act(obs, critic_obs)
        self.amp_transition.observations = amp_obs
        return actions

    def process_env_step(self, rewards, dones, infos, amp_next_obs):
        """Store environment step for PPO and AMP transitions for GAIL."""
        # Insert AMP transition before PPO clears its transition
        self.amp_storage.insert(self.amp_transition.observations, amp_next_obs)
        super().process_env_step(rewards, dones, infos)
        self.amp_transition.clear()

    @torch.no_grad()
    def compute_returns(self, last_critic_obs):
        """Delegate to base PPO implementation."""
        return super().compute_returns(last_critic_obs)

    def _safe_gail_reward(self, logits: torch.Tensor) -> torch.Tensor:
        """Numerically stable GAIL reward: -log(1 - D(s,a)) via logits."""
        return -F.logsigmoid(-logits)

    @timeit("update_time")
    def update(self) -> Dict[str, float]:
        """Perform one GAIL + PPO update over the collected rollout."""
        if self.storage is None:
            return {}

        if self.actor.training is False:
            self.train_mode()

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

        mean_val_loss = 0.0
        mean_surr_loss = 0.0
        mean_disc_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        updates = 0

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

            # ---------------- Discriminator update ---------------- #
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize(policy_state)
                    policy_next_state = self.amp_normalizer.normalize(policy_next_state)
                    expert_state = self.amp_normalizer.normalize(expert_state)
                    expert_next_state = self.amp_normalizer.normalize(expert_next_state)

            for _ in range(self.cfg.disc_steps):
                policy_logits = self.discriminator(
                    torch.cat([policy_state, policy_next_state], dim=-1)
                )
                expert_logits = self.discriminator(
                    torch.cat([expert_state, expert_next_state], dim=-1)
                )

                expert_labels = torch.ones_like(expert_logits, device=self.device)
                policy_labels = torch.zeros_like(policy_logits, device=self.device)

                expert_loss = self.bce_loss(expert_logits, expert_labels)
                policy_loss = self.bce_loss(policy_logits, policy_labels)
                d_loss = 0.5 * (expert_loss + policy_loss)

                if hasattr(self.discriminator, "compute_grad_pen"):
                    gp = self.discriminator.compute_grad_pen(
                        expert_state, expert_next_state, lambda_=self.cfg.grad_pen_lambda
                    )
                    d_loss = d_loss + gp

                self.disc_optimizer.zero_grad()
                d_loss.backward()
                self.disc_optimizer.step()

            mean_disc_loss += d_loss.item()
            mean_policy_pred += torch.sigmoid(policy_logits).mean().item()
            mean_expert_pred += torch.sigmoid(expert_logits).mean().item()

            # ---------------- PPO update with shaped rewards ---------------- #
            self.actor.act(obs_batch)
            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            value_batch = self.critic(critic_obs_batch)

            mu_batch = self.actor.action_mean
            sigma_batch = self.actor.action_std
            entropy_batch = self.actor.entropy()

            # Adaptive KL (optional)
            if self.cfg.desired_kl is not None and self.cfg.schedule == "adaptive":
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
                        for pg in self.policy_optimizer.param_groups:
                            pg["lr"] = max(1e-5, pg["lr"] / 1.5)
                    elif self.cfg.desired_kl / 2.0 < kl_mean:
                        for pg in self.policy_optimizer.param_groups:
                            pg["lr"] = min(1e-2, pg["lr"] * 1.5)

            with torch.no_grad():
                policy_logits_for_reward = self.discriminator(
                    torch.cat([policy_state, policy_next_state], dim=-1)
                )
                gail_r = self._safe_gail_reward(policy_logits_for_reward).squeeze()
                if returns_batch.dim() > gail_r.dim():
                    gail_r = gail_r.unsqueeze(1)
                returns_batch = returns_batch + self.cfg.lambda_gail * gail_r
                advantages_batch = advantages_batch + self.cfg.lambda_gail * gail_r

            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.cfg.clip_param,
                1.0 + self.cfg.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.cfg.use_clipped_value_loss:
                value_clipped = old_values_batch + (value_batch - old_values_batch).clamp(
                    -self.cfg.clip_param, self.cfg.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            total_policy_loss = (
                surrogate_loss
                + self.cfg.value_loss_coef * value_loss
                - self.cfg.entropy_coef * entropy_batch.mean()
            )

            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.cfg.max_grad_norm,
            )
            self.policy_optimizer.step()

            if self.min_std is not None:
                self.actor.action_std.data.clamp_(min=self.min_std)

            mean_val_loss += value_loss.item()
            mean_surr_loss += surrogate_loss.item()
            updates += 1

        if updates == 0:
            return {}

        denom = max(1, self.cfg.num_learning_epochs * self.cfg.num_mini_batches)

        self.storage.clear()
        return {
            "mean_value_loss": mean_val_loss / updates,
            "mean_surrogate_loss": mean_surr_loss / updates,
            "mean_disc_loss": mean_disc_loss / denom,
            "mean_policy_pred": mean_policy_pred / denom,
            "mean_expert_pred": mean_expert_pred / denom,
        }


@configclass
class GAILPPOCfg(PPOCfg):
    """Configuration for GAILPPO algorithm."""

    class_type: type[GAILPPO] = GAILPPO

    # GAIL-specific (in addition to standard PPO settings)
    lambda_gail: float = 1.0
    disc_steps: int = 1
    grad_pen_lambda: float = 10.0
    policy_lr: float = 3e-4
    disc_lr: float = 3e-4
    min_std: float | None = None

    # Runtime provided
    amp_obs_dim: int = MISSING
    motion_dataset: object = MISSING

    def construct_from_cfg(
        self,
        actor_critic,
        discriminator: Discriminator,
        amp_normalizer: RunningMeanStd | None,
        motion_dataset,
        amp_obs_dim: int,
        device,
        *args,
        **kwargs,
    ):
        """Custom factory for GAILPPO with additional dependencies."""
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


__all__ = ["GAILPPO", "GAILPPOCfg"]

