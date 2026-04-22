"""
D4RL Environment Wrapper for RoboRenForce

Directly wraps D4RL offline RL benchmark behind RoboRenForceVecEnv.
No dependency on RLinf — uses upstream d4rl + gymnasium directly.

D4RL is state-only (no images) — uses classic VecEnv interface.

Supported tasks:
    walker2d-medium-v2, walker2d-medium-replay-v2, walker2d-medium-expert-v2
    hopper-medium-v2, hopper-medium-replay-v2, hopper-medium-expert-v2
    halfcheetah-medium-v2, halfcheetah-medium-replay-v2, halfcheetah-medium-expert-v2

Observation: [B, state_dim] flat state vector
Action: [B, action_dim] continuous control

Dependencies:
    pip install d4rl gymnasium[mujoco]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from RoboRenForce.prototype.classic.vec_env import RoboRenForceVecEnv


@dataclass
class D4RLTaskConfig:
    """Configuration for D4RL environment."""
    task_name: str = "walker2d-medium-v2"
    action_dim: int = 6
    state_dim: int = 17
    max_episode_steps: int = 1000
    seed: int = 0


class D4RLRRFEnv(RoboRenForceVecEnv):
    """D4RL environment implementing RoboRenForceVecEnv.

    Directly uses upstream d4rl + gym packages (no RLinf dependency).
    D4RL is state-only, appropriate for offline RL (IQL, CQL, TD3+BC).

    Usage:
        cfg = {"task_name": "walker2d-medium-v2", "state_dim": 17, "action_dim": 6}
        env = D4RLRRFEnv(cfg, num_envs=1, device="cuda:0")
        obs, extras = env.reset()
        obs, rew, dones, extras = env.step(actions)
        dataset = env.get_dataset()  # for offline RL
    """

    def __init__(self, cfg: dict, num_envs: int = 1, device: str = "cpu", **kwargs):
        defaults = D4RLTaskConfig()
        for k, v in defaults.__dict__.items():
            cfg.setdefault(k, v)

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.num_obs = cfg["state_dim"]
        self.num_actions = cfg["action_dim"]
        self.num_privileged_obs = 0
        self.max_episode_length = cfg.get("max_episode_steps", 1000)

        # Buffers
        self.obs_buf = torch.zeros(num_envs, self.num_obs, device=self.device)
        self.rew_buf = torch.zeros(num_envs, device=self.device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.privileged_obs_buf = None
        self.extras = {}

        self._envs = []
        self._init_sim(cfg, num_envs)

    def _init_sim(self, cfg: dict, num_envs: int):
        """Create D4RL gym environments."""
        try:
            import gym
            import d4rl  # noqa: F401  — registers d4rl envs
        except ImportError:
            raise ImportError(
                "D4RL not found. Install it:\n"
                "  pip install d4rl gymnasium[mujoco]"
            )

        task_name = cfg["task_name"]
        for i in range(num_envs):
            env = gym.make(task_name)
            env.seed(cfg.get("seed", 0) + i)
            self._envs.append(env)

    def _obs_to_tensor(self, obs_list: list[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack(obs_list)).float().to(self.device)

    # ---- VecEnv interface ----

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        return self.obs_buf, {"observations": {"policy": self.obs_buf}}

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_list = [env.reset() for env in self._envs]
        self.obs_buf = self._obs_to_tensor(obs_list)
        self.episode_length_buf.zero_()
        self.extras = {"observations": {"policy": self.obs_buf}}
        return self.obs_buf, self.extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy()
        obs_list, rew_list, done_list, term_list, trunc_list = [], [], [], [], []

        for i, env in enumerate(self._envs):
            result = env.step(actions_np[i])
            if len(result) == 5:
                obs, rew, terminated, truncated, info = result
            else:
                obs, rew, done, info = result
                terminated = done and not info.get("TimeLimit.truncated", False)
                truncated = info.get("TimeLimit.truncated", False)

            # Auto-reset
            if terminated or truncated:
                obs = env.reset()

            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(terminated)
            trunc_list.append(truncated)

        self.obs_buf = self._obs_to_tensor(obs_list)
        self.rew_buf = torch.tensor(rew_list, dtype=torch.float32, device=self.device)
        terminated_t = torch.tensor(term_list, dtype=torch.bool, device=self.device)
        truncated_t = torch.tensor(trunc_list, dtype=torch.bool, device=self.device)
        dones = (terminated_t | truncated_t).long()
        self.reset_buf = dones

        self.episode_length_buf += 1
        # Reset counter for done envs
        self.episode_length_buf[dones.bool()] = 0

        self.extras = {
            "observations": {"policy": self.obs_buf},
            "termination": terminated_t,
            "timeout": truncated_t,
        }
        return self.obs_buf, self.rew_buf, dones, self.extras

    def get_dataset(self) -> dict:
        """Return the offline D4RL dataset for offline RL training."""
        if hasattr(self._envs[0], "get_dataset"):
            return self._envs[0].get_dataset()
        raise RuntimeError(f"Cannot load D4RL dataset for {self.cfg['task_name']}")

    def get_normalized_score(self, returns: float) -> float:
        """Return D4RL normalized score (0-100 scale)."""
        if hasattr(self._envs[0], "get_normalized_score"):
            return float(self._envs[0].get_normalized_score(returns)) * 100.0
        return returns

    def close(self):
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()
        self._envs = []
