"""Dynamic env wrapper for MJLab — adds reward/command extraction and dim_params."""

import torch
from .vecenv_wrapper import RoboRenForceMJLabEnvWrapper


class MJLabDynamicEnvWrapper(RoboRenForceMJLabEnvWrapper):
    """Extends MJLab wrapper with reward/command access and dim_params property.

    Mirrors RFDynamicEnvWrapper but for MJLab environments.
    """

    def __init__(self, env, clip_actions=None):
        super().__init__(env, clip_actions)
        self.rewards_shape = self.get_rewards().shape[-1]
        self.command_shape = self.get_commands().shape[-1]

    def get_rewards(self):
        return self.unwrapped.reward_manager._step_reward

    def get_commands(self):
        commands = []
        for k, v in self.unwrapped.command_manager._terms.items():
            commands.append(v.command)
        if commands:
            return torch.cat(commands, dim=1)
        else:
            return torch.zeros((self.num_envs, 0), device=self.device)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs_dict = self.unwrapped.observation_manager.compute()
        obs_dict = self._remap_obs(obs_dict)
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs_dict = self._remap_obs(obs_dict)

        dones = (terminated | truncated).to(dtype=torch.long)
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        extras["termination"] = terminated
        extras["command"] = self.get_commands()

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
            extras["timeout"] = truncated

        return obs, rew, dones, extras

    @property
    def dim_params(self):
        dim_params = {k + "_dim": v for k, v in self.observation_space.items()}
        _dim_params = {
            "policy_dim": self.observation_space["policy"].shape[-1],
            "critic_dim": self.observation_space.get("critic", self.observation_space["policy"]).shape[-1],
            "dynamic_dim": self.observation_space.get("dynamic", self.observation_space["policy"]).shape[-1],
            "action_dim": self.action_space.shape[-1],
            "rewards_dim": self.rewards_shape,
        }
        dim_params.update(_dim_params)
        return dim_params
