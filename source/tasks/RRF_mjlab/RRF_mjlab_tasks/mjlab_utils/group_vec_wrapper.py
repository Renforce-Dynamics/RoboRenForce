"""Group vec wrapper for MJLab — train/eval env partitioning."""

import torch
from .dynamic_env_wrapper import MJLabDynamicEnvWrapper


class MJLabGroupVecWrapper(MJLabDynamicEnvWrapper):
    """Partitions MJLab environment into training and evaluation groups.

    Mirrors GroupVecWrapper but for MJLab environments.
    Externally exposed num_envs equals num_train_envs.
    """

    def __init__(self, env, clip_actions=None, *, num_eval_envs=None):
        super().__init__(env, clip_actions)

        if num_eval_envs is None:
            num_eval_envs = int(0.2 * self.num_envs)

        self.total_envs = self.num_envs
        self.num_eval_envs = num_eval_envs
        self.num_train_envs = self.total_envs - self.num_eval_envs
        self.num_envs = self.num_train_envs

    def get_observations(self):
        obs_dict = self.unwrapped.observation_manager.compute()
        obs_dict = self._remap_obs(obs_dict)
        return obs_dict["policy"], {"observations": obs_dict}

    def reset(self):
        obs_dict, _ = self.env.reset()
        obs_dict = self._remap_obs(obs_dict)
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions, *, eval_actions=None):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        if actions.shape[0] != self.num_train_envs:
            raise RuntimeError(
                f"Action batch ({actions.shape[0]}) does not match "
                f"num_train_envs ({self.num_train_envs})."
            )

        if self.num_eval_envs > 0:
            if eval_actions is None:
                eval_actions = torch.zeros(
                    (self.num_eval_envs, actions.shape[1]),
                    dtype=actions.dtype,
                    device=actions.device,
                )
            full_actions = torch.cat([actions, eval_actions], dim=0)
        else:
            full_actions = actions

        obs_dict, rew, terminated, truncated, extras = self.env.step(full_actions)
        obs_dict = self._remap_obs(obs_dict)

        dones = (terminated | truncated).long()
        obs = obs_dict["policy"]

        extras["observations"] = obs_dict
        extras["termination"] = terminated
        extras["rewards"] = self.unwrapped.reward_manager._step_reward

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["timeout"] = truncated

        return obs, rew, dones, extras

    # -- Split utilities --

    def _split_tensor(self, x: torch.Tensor):
        t = self.num_train_envs
        return x[:t], x[t:]

    def _split_dict(self, d: dict):
        train_d, eval_d = {}, {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                train_d[k], eval_d[k] = self._split_tensor(v)
            else:
                train_d[k] = v
                eval_d[k] = v
        return train_d, eval_d

    def separate_observations(self, obs_tensor, obs_info):
        train_obs, eval_obs = self._split_tensor(obs_tensor)
        train_info, eval_info = self._split_dict(obs_info)
        return (train_obs, train_info), (eval_obs, eval_info)

    def separate_step(self, obs, rew, dones, extras):
        (train_obs, train_info), (eval_obs, eval_info) = self.separate_observations(
            obs, extras["observations"]
        )
        train_rew, eval_rew = self._split_tensor(rew)
        train_done, eval_done = self._split_tensor(dones)
        train_extra, eval_extra = self._split_dict(extras)
        train_extra["observations"] = train_info
        eval_extra["observations"] = eval_info
        return (train_obs, train_rew, train_done, train_extra), \
               (eval_obs, eval_rew, eval_done, eval_extra)

    def separate_dict(self, d):
        return self._split_dict(d)
