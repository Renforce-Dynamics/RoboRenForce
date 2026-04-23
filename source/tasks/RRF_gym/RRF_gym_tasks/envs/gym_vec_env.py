"""
Gymnasium → RoboRenForce VecEnv wrapper.

Wraps a single Gymnasium continuous-action env as a RoboRenForceVecEnv
(num_envs=1). Handles auto-reset, obs/reward tensor conversion, and
the extras dict expected by OffPolicyRunner / OnPolicyRunner.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from RoboRenForce.prototype.classic import RoboRenForceVecEnv


class GymVecEnv(RoboRenForceVecEnv):
    """Wraps a single Gymnasium env into the RoboRenForce VecEnv interface."""

    def __init__(self, task_name: str, device: str = "cpu", seed: int = 42):
        self.env = gym.make(task_name)
        self.env.reset(seed=seed)
        self.device = torch.device(device)
        self._task_name = task_name

        # Dimensions
        self.num_envs = 1
        self.num_obs = gym.spaces.flatdim(self.env.observation_space)
        self.num_actions = gym.spaces.flatdim(self.env.action_space)
        self.num_privileged_obs = 0
        self.max_episode_length = self.env.spec.max_episode_steps or 1000

        # Buffers
        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)
        self._obs = None

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def get_observations(self):
        if self._obs is None:
            return self.reset()
        return self._obs, self._make_obs_extras(self._obs)

    def reset(self):
        obs_np, _ = self.env.reset()
        self.episode_length_buf.zero_()
        self._obs = self._to_tensor(obs_np)
        return self._obs, self._make_obs_extras(self._obs)

    def step(self, actions: torch.Tensor):
        act_np = actions.detach().cpu().numpy().flatten()
        obs_np, reward, terminated, truncated, _ = self.env.step(act_np)

        done = terminated or truncated
        self.episode_length_buf += 1

        # td_obs is the *real* next obs (before auto-reset)
        td_obs = self._to_tensor(obs_np)

        # Auto-reset
        if done:
            obs_np, _ = self.env.reset()
            self.episode_length_buf.zero_()
        self._obs = self._to_tensor(obs_np)

        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([done], dtype=torch.long, device=self.device)

        extras = {
            "td_observations": {"policy": td_obs, "critic": td_obs},
            "observations": {"policy": self._obs, "critic": self._obs},
            "termination": torch.tensor([terminated], dtype=torch.float32, device=self.device),
            "timeout": torch.tensor([truncated], dtype=torch.float32, device=self.device),
        }

        return self._obs, reward_t, done_t, extras

    def close(self):
        self.env.close()

    def seed(self, seed: int):
        self.env.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def dim_params(self):
        return {
            "policy_dim": self.num_obs,
            "critic_dim": self.num_obs,
            "action_dim": self.num_actions,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _to_tensor(self, obs_np) -> torch.Tensor:
        return torch.as_tensor(
            obs_np, dtype=torch.float32, device=self.device
        ).view(1, -1)

    def _make_obs_extras(self, obs: torch.Tensor) -> dict:
        return {"observations": {"policy": obs, "critic": obs}}
