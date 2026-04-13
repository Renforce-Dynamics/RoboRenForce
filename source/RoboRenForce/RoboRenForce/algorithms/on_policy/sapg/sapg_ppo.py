from __future__ import annotations

import torch

from RoboRenForce import configclass

from RoboRenForce.algorithms.on_policy.ppo import PPO, PPOCfg
from RoboRenForce.buffer.online_rollout.sapg_rollout_storage import SAPGRolloutStorage
from RoboRenForce.algorithms.algorithm_base import AlgorithmBaseCfg


class SAPGPPO(PPO):
    """
    PPO variant with SAPG (Split and Aggregate Policy Gradients) support.

    Differences from vanilla PPO:
    - Uses `SAPGRolloutStorage` instead of `RolloutStorage`.
    - `update` accepts an optional `batch_augmenter` and, when provided,
      runs SAPG batch augmentation lazily inside the mini-batch generator.
    """

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
        use_sapg: bool = True,
    ):
        """
        Initialize rollout storage.

        Args:
            num_envs: Number of parallel environments
            num_transitions_per_env: Horizon per environment
            actor_obs_shape: Shape of actor observations
            critic_obs_shape: Shape of critic observations
            action_shape: Shape of actions
            use_sapg: Kept for API compatibility; always uses `SAPGRolloutStorage`.
        """
        self.storage = SAPGRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def update(self, batch_augmenter=None):
        """
        Perform a PPO update with optional SAPG batch augmentation.

        If `batch_augmenter` is provided, `SAPGRolloutStorage.mini_batch_generator`
        will be used, which internally:
        - runs SAPG augmentation to build an augmented flat batch;
        - shuffles and splits it into mini-batches;
        - optionally returns an `off_policy_mask` per mini-batch (currently unused
          in the loss).
        """
        from RoboRenForce.utils.logging import timeit  # local import to avoid cycle

        # NOTE: We keep the same logging decorator behaviour by reusing PPO's
        # `update` decorator logic. The actual timing is already handled by
        # the outer @timeit on PPO.update, so we do not re-decorate here.

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0

        # Select appropriate mini-batch generator
        if batch_augmenter is not None:
            generator = self.storage.sapg_mini_batch_generator(
                batch_augmenter=batch_augmenter,
                critic=self.critic,
                num_mini_batches=self.cfg.num_mini_batches,
                num_epochs=self.cfg.num_learning_epochs,
            )
        else:
            # Fallback to standard generator if no augmenter is provided
            generator = self.storage.mini_batch_generator(
                self.cfg.num_mini_batches, self.cfg.num_learning_epochs
            )

        for batch in generator:
            # SAPG-augmented batches include an extra `off_policy_mask` element.
            if len(batch) == 10:
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
                    off_policy_mask_batch,  # noqa: F841 - currently unused
                ) = batch
            else:
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

            # Standard PPO loss (same as PPO.update)
            self.actor.act(obs_batch)
            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            entropy = self.actor.entropy()
            value_pred = self.critic(critic_obs_batch)
            mu_batch = self.actor.action_mean
            sigma_batch = self.actor.action_std

            # Adaptive KL (unchanged)
            if self.cfg.schedule == "adaptive" and self.cfg.desired_kl is not None:
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.cfg.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
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

            # Value loss
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

            loss = (
                surrogate_loss
                + self.cfg.value_loss_coef * value_loss
                - self.cfg.entropy_coef * entropy.mean()
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        self.storage.clear()

        return {
            "mean_value_loss": mean_value_loss / num_updates,
            "mean_surrogate_loss": mean_surrogate_loss / num_updates,
            "learning_rate": self.learning_rate,
            "mean_std": self.actor.action_std.mean(),
        }


@configclass
class SAPGPPOCfg(PPOCfg):
    """
    Configuration for SAPGPPO.

    Inherits all fields from `PPOCfg` but changes `class_type` to `SAPGPPO`.
    """

    class_type: type["SAPGPPO"] = SAPGPPO

