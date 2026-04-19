from __future__ import annotations

import torch

from RRF_isaaclab_tasks.env_wrapper.dynamic_env_wrapper import RFDynamicEnvWrapper


class AMPEnvWrapper(RFDynamicEnvWrapper):
    """Environment wrapper for AMP-style adversarial imitation.

    This wrapper extends the standard dynamic env wrapper by:
    - Exposing AMP observations used by the discriminator.
    - Returning terminal AMP states on reset for proper handling of episode ends.
    """

    def __init__(self, env, clip_actions=None, *, motion_dataset=None):
        super().__init__(env, clip_actions)
        # Expose motion dataset on the underlying environment for convenience.
        self.unwrapped.motion_dataset = motion_dataset

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Return policy observations and full observation dict as extras."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    def get_amp_observations(self) -> torch.Tensor:
        """Return AMP observations used for adversarial training."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["amp"]

    def step(self, actions: torch.Tensor, *, not_amp: bool = True, **kwargs):
        """Environment step with optional AMP information.

        When ``not_amp`` is False, this method returns additional AMP-related
        information used by adversarial algorithms:

        Returns:
            obs: policy observations
            privileged_obs: critic / privileged observations (if available)
            rewards: task rewards
            dones: done flags
            extras: additional information dict
            reset_env_ids: indices of environments that were reset
            terminal_amp_states: AMP observations at terminal states
        """
        if not_amp:
            return super().step(actions)

        # Clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # Step underlying environment (IsaacLab style API)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # Build done flags and observation views
        dones = (terminated | truncated).to(dtype=torch.long)
        obs = obs_dict["policy"].clamp(-500, 500)
        privileged_obs = obs_dict.get("critic", obs).clamp(-500, 500)
        amp_obs = obs_dict.get("amp", obs).clamp(-500, 500)

        extras["observations"] = obs_dict
        extras["termination"] = terminated

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
            extras["timeout"] = truncated

        reset_env_ids = torch.where(dones)[0]
        terminal_amp_states = amp_obs[reset_env_ids]

        return (
            obs,
            privileged_obs,
            rew,
            dones,
            extras,
            reset_env_ids,
            terminal_amp_states,
        )

