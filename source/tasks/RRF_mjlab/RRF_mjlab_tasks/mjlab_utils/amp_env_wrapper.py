"""AMP env wrapper for MJLab.

Mirrors :class:`RRF_isaaclab_tasks.env_wrapper.adversarial_wrapper.AMPEnvWrapper`
but targets the MJLab backend on top of :class:`MJLabDynamicEnvWrapper`.

Reference: beyondAMP/source/beyondAMP/beyondAMP/mjlab/rsl_rl/amp_wrapper.py
"""

from __future__ import annotations

import torch

from .dynamic_env_wrapper import MJLabDynamicEnvWrapper

_AMP_GROUP = "amp"


class MJLabAMPEnvWrapper(MJLabDynamicEnvWrapper):
    """MJLab AMP wrapper.

    Adds an ``amp`` observation group on top of MJLabDynamicEnvWrapper, an
    AMP-aware step that returns the seven-tuple expected by
    :class:`RoboRenForce.runners.imitation.adversarial.AMPOnPolicyImitationRunner`,
    and exposes :attr:`motion_dataset` on the underlying env.
    """

    def __init__(
        self,
        env,
        clip_actions: float | None = None,
        *,
        motion_dataset=None,
        amp_group: str = _AMP_GROUP,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)
        self._amp_group = amp_group
        self.motion_dataset = motion_dataset
        # Stash on underlying env so MDP terms (rewards / events) can reach it.
        self.unwrapped.motion_dataset = self.motion_dataset

    def get_amp_observations(self) -> torch.Tensor:
        obs_dict = self.unwrapped.observation_manager.compute()
        return obs_dict[self._amp_group]

    def step(self, actions: torch.Tensor, *, not_amp: bool = True, **kwargs):
        if not_amp:
            return super().step(actions)

        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs_dict = self._remap_obs(obs_dict)

        dones = (terminated | truncated).to(dtype=torch.long)
        obs = obs_dict["policy"].clamp(-500, 500)
        privileged_obs = obs_dict.get("critic", obs).clamp(-500, 500)
        amp_obs = obs_dict.get(self._amp_group, obs).clamp(-500, 500)

        extras["observations"] = obs_dict
        extras["termination"] = terminated
        extras["command"] = self.get_commands()

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

    @property
    def dof_pos_limits(self) -> torch.Tensor:
        return self.unwrapped.scene["robot"].data.joint_pos_limits
