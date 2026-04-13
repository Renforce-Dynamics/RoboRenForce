from __future__ import annotations

from typing import Dict, Callable, Iterator, Tuple, Optional

import torch

from RoboRenForce.buffer.online_rollout.rollout_storage import RolloutStorage


class SAPGRolloutStorage(RolloutStorage):
    """
    Rollout storage with built-in SAPG batch augmentation support.

    Design goals:
    - Reuse the existing `RolloutStorage` data layout and write logic;
    - Perform SAPG augmentation lazily at mini-batch generation time without
      changing the underlying storage tensor sizes;
    - Expose `sapg_mini_batch_generator` so the algorithm can directly consume
      batches that include off-policy samples.
    """

    @torch.no_grad()
    def _build_sapg_extras(
        self,
        critic: torch.nn.Module,
    ) -> Dict:
        """
        Build the `extras` dictionary required by SAPG augmentation
        from the data currently stored in this rollout buffer.

        This mirrors the extras construction logic used in
        `SAPGOnPolicyRunner._augment_storage_before_update`.
        """
        # Storage format: [horizon, num_envs, ...]
        horizon = int(self.step)
        num_envs = int(self.num_envs)

        if horizon == 0:
            return {
                "obs": None,
                "rewards": None,
                "dones": None,
                "last_obs": {"obs": None},
                "last_dones": None,
                "last_values": None,
            }

        obs = self.observations[:horizon]  # [horizon, num_envs, obs_dim]
        if getattr(self, "privileged_observations", None) is not None:
            critic_obs = self.privileged_observations[:horizon]
        else:
            critic_obs = obs

        extras: Dict[str, torch.Tensor] = {
            "obs": obs,
            "rewards": self.rewards[:horizon],
            "dones": self.dones[:horizon],
            "last_obs": {"obs": None},
            "last_dones": None,
            "last_values": None,
        }

        # 末步观测与 bootstrap value
        last_obs = obs[-1]  # [num_envs, obs_dim]
        last_critic_obs = critic_obs[-1]  # [num_envs, obs_dim]
        extras["last_obs"]["obs"] = last_obs
        extras["last_dones"] = self.dones[horizon - 1]  # [num_envs, 1]

        last_values = critic(last_critic_obs)
        extras["last_values"] = last_values  # [num_envs, 1]

        return extras

    @torch.no_grad()
    def _build_sapg_batch_dict(self) -> Dict[str, torch.Tensor]:
        """
        Build a flattened `batch_dict` for SAPG augmentation from storage.

        Returns a dict with shapes like:
        - 'returns':  [T * N, 1]
        - 'values':   [T * N, 1]
        - 'obses':    [T * N, obs_dim]
        - Optional keys ('actions_log_prob', 'mu', 'sigma', etc.) are added if present.
        """
        horizon = int(self.step)
        num_envs = int(self.num_envs)

        if horizon == 0:
            raise RuntimeError("SAPGRolloutStorage: no data in storage (step == 0).")

        obs = self.observations[:horizon]  # [T, N, obs_dim]

        batch_dict: Dict[str, torch.Tensor] = {
            "returns": self.returns[:horizon].reshape(horizon * num_envs, -1),
            "values": self.values[:horizon].reshape(horizon * num_envs, -1),
            "obses": obs.reshape(horizon * num_envs, -1),
        }

        # 可选字段
        if hasattr(self, "actions_log_prob") and self.actions_log_prob is not None:
            batch_dict["actions_log_prob"] = self.actions_log_prob[:horizon].reshape(
                horizon * num_envs, -1
            )
        if hasattr(self, "mu") and self.mu is not None:
            batch_dict["mu"] = self.mu[:horizon].reshape(horizon * num_envs, -1)
        if hasattr(self, "sigma") and self.sigma is not None:
            batch_dict["sigma"] = self.sigma[:horizon].reshape(horizon * num_envs, -1)

        return batch_dict

    @torch.no_grad()
    def build_sapg_augmented_batch(
        self,
        batch_augmenter,
        critic: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a single SAPG augmentation pass and return a flattened augmented batch dict.

        This method does not mutate the underlying storage; it only computes
        augmented returns/values (and derived advantages) from the current rollout.
        """

        def get_values_fn(obs_batch: torch.Tensor) -> torch.Tensor:
            return critic(obs_batch)

        extras = self._build_sapg_extras(critic)
        batch_dict = self._build_sapg_batch_dict()

        augmented_batch = batch_augmenter.augment_batch(
            batch_dict=batch_dict,
            extras=extras,
            get_values_fn=get_values_fn,
        )

        # Recompute advantages from the new returns / values
        returns_flat = augmented_batch["returns"]
        values_flat = augmented_batch["values"]
        advantages_flat = returns_flat - values_flat
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )
        augmented_batch["advantages"] = advantages_flat

        return augmented_batch

    @torch.no_grad()
    def sapg_mini_batch_generator(
        self,
        batch_augmenter,
        critic: torch.nn.Module,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Iterator[
        Tuple[
            torch.Tensor,  # obs_batch
            torch.Tensor,  # critic_obs_batch
            torch.Tensor,  # actions_batch
            torch.Tensor,  # old_values_batch
            torch.Tensor,  # advantages_batch
            torch.Tensor,  # returns_batch
            torch.Tensor,  # old_actions_log_prob_batch
            torch.Tensor,  # old_mu_batch
            torch.Tensor,  # old_sigma_batch
            Optional[torch.Tensor],  # off_policy_mask_batch
        ]
    ]:
        """
        Mini-batch generator with SAPG augmentation.

        - First calls `SAPGBatchAugmenter` on the current rollout to obtain
          an augmented flat batch;
        - Then shuffles and splits that flat batch into mini-batches;
        - Optionally returns an `off_policy_mask` per mini-batch so the loss
          function can distinguish on-/off-policy samples if desired.
        """
        augmented_batch = self.build_sapg_augmented_batch(
            batch_augmenter=batch_augmenter,
            critic=critic,
        )

        # Total number of samples after flattening/augmentation
        total_size = augmented_batch["returns"].shape[0]
        batch_size = total_size
        mini_batch_size = batch_size // num_mini_batches

        device = augmented_batch["returns"].device
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=device,
        )

        # Observations used by both actor and critic. For now we share the same
        # tensor; if we need separate actor/critic observations in the future,
        # we can extend `augment_batch` to return extra keys.
        if "obses" in augmented_batch:
            observations = augmented_batch["obses"]
        else:
            # Fallback: directly use the raw storage observations
            # (this should only be hit in edge cases).
            horizon = int(self.step)
            observations = self.observations[:horizon].reshape(horizon * self.num_envs, -1)
            # If augmented is longer than the original, simply repeat
            # to match sizes (consistent with `SAPGBatchAugmenter._augment_tensor`).
            repeat_factor = total_size // observations.shape[0]
            observations = torch.cat([observations] * repeat_factor, dim=0)

        actions = self.actions[: self.step].reshape(self.step * self.num_envs, -1)
        # Repeat actions to match the augmented length (same as observations)
        repeat_factor = total_size // actions.shape[0]
        actions = torch.cat([actions] * repeat_factor, dim=0)

        values = augmented_batch["values"]
        returns = augmented_batch["returns"]
        advantages = augmented_batch["advantages"]

        old_actions_log_prob = augmented_batch.get(
            "actions_log_prob",
            self.actions_log_prob[: self.step].reshape(self.step * self.num_envs, -1),
        )
        if old_actions_log_prob.shape[0] != total_size:
            repeat_factor = total_size // old_actions_log_prob.shape[0]
            old_actions_log_prob = torch.cat(
                [old_actions_log_prob] * repeat_factor, dim=0
            )

        old_mu = augmented_batch.get(
            "mu",
            self.mu[: self.step].reshape(self.step * self.num_envs, -1),
        )
        if old_mu.shape[0] != total_size:
            repeat_factor = total_size // old_mu.shape[0]
            old_mu = torch.cat([old_mu] * repeat_factor, dim=0)

        old_sigma = augmented_batch.get(
            "sigma",
            self.sigma[: self.step].reshape(self.step * self.num_envs, -1),
        )
        if old_sigma.shape[0] != total_size:
            repeat_factor = total_size // old_sigma.shape[0]
            old_sigma = torch.cat([old_sigma] * repeat_factor, dim=0)

        off_policy_mask = augmented_batch.get("off_policy_mask", None)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_obs_batch = obs_batch  # actor / critic share observations in this implementation
                actions_batch = actions[batch_idx]
                old_values_batch = values[batch_idx]
                advantages_batch = advantages[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                off_policy_mask_batch = (
                    off_policy_mask[batch_idx] if off_policy_mask is not None else None
                )

                yield (
                    obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    old_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    off_policy_mask_batch,
                )

