"""
D4RL Environment Wrapper for RoboRenForce

Wraps RLinf's D4RLEnv behind the RoboRenForceVecEnv interface.

D4RL is a state-only offline RL benchmark — no images, no task descriptions.
This wrapper uses the classic VecEnv interface (tensor obs) rather than
EmbodiedEnv (multimodal obs).

Supported tasks:
    walker2d-medium-v2, walker2d-medium-replay-v2, walker2d-medium-expert-v2
    hopper-medium-v2, hopper-medium-replay-v2, hopper-medium-expert-v2
    halfcheetah-medium-v2, halfcheetah-medium-replay-v2, halfcheetah-medium-expert-v2

Observation format (from RLinf):
    states: [B, state_dim] - MuJoCo proprioception

Action format: [B, action_dim] - continuous control
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

    # RLinf-specific
    seed: int = 0
    auto_reset: bool = True
    use_subproc: bool = False


class D4RLRRFEnv(RoboRenForceVecEnv):
    """D4RL environment implementing RoboRenForceVecEnv.

    D4RL is state-only (no images), so we use the classic VecEnv interface
    rather than EmbodiedEnv. This is appropriate for offline RL algorithms
    like IQL, CQL, TD3+BC.

    Usage:
        cfg = {
            "task_name": "walker2d-medium-v2",
            "state_dim": 17,
            "action_dim": 6,
            "max_episode_steps": 1000,
        }
        env = D4RLRRFEnv(cfg, num_envs=1, device="cuda:0")
        obs, extras = env.reset()
        obs, rew, dones, extras = env.step(actions)
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

        self._rlinf_env = None
        self._init_rlinf_env(cfg, num_envs, kwargs)

    def _init_rlinf_env(self, cfg: dict, num_envs: int, kwargs: dict):
        """Construct the underlying RLinf D4RLEnv."""
        try:
            from omegaconf import OmegaConf
            from rlinf.envs.d4rl.d4rl_env import D4RLEnv
        except ImportError:
            raise ImportError(
                "D4RL environment requires rlinf and d4rl:\n"
                "  pip install rlinf\n"
                "  pip install d4rl gymnasium[mujoco]"
            )

        rlinf_cfg = OmegaConf.create({
            "task_name": cfg["task_name"],
            "seed": cfg.get("seed", 0),
            "auto_reset": cfg.get("auto_reset", True),
            "use_subproc": cfg.get("use_subproc", False),
            "max_episode_steps": cfg.get("max_episode_steps", 1000),
        })

        seed_offset = kwargs.get("seed_offset", 0)
        total_num_processes = kwargs.get("total_num_processes", 1)
        worker_info = kwargs.get("worker_info", None)

        self._rlinf_env = D4RLEnv(
            cfg=rlinf_cfg,
            num_envs=num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=worker_info,
        )

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs, infos = self._rlinf_env.reset()
        states = obs.get("states", obs.get("state"))
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        self.obs_buf = states.to(self.device)
        return self.obs_buf, {"observations": {"policy": self.obs_buf}}

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, infos = self._rlinf_env.reset()
        states = obs.get("states", obs.get("state"))
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        self.obs_buf = states.to(self.device)
        self.episode_length_buf.zero_()
        self.extras = {"observations": {"policy": self.obs_buf}, "infos": infos}
        return self.obs_buf, self.extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy()
        obs, rewards, terminated, truncated, infos = self._rlinf_env.step(actions_np)

        states = obs.get("states", obs.get("state"))
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        self.obs_buf = states.to(self.device)

        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).float()
        self.rew_buf = rewards.to(self.device)

        if isinstance(terminated, np.ndarray):
            terminated = torch.from_numpy(terminated)
        if isinstance(truncated, np.ndarray):
            truncated = torch.from_numpy(truncated)
        dones = (terminated | truncated).long().to(self.device)
        self.reset_buf = dones

        self.episode_length_buf += 1
        self.extras = {
            "observations": {"policy": self.obs_buf},
            "termination": terminated.to(self.device),
            "timeout": truncated.to(self.device),
            "infos": infos,
        }
        return self.obs_buf, self.rew_buf, dones, self.extras

    def close(self):
        if self._rlinf_env is not None and hasattr(self._rlinf_env, "close"):
            self._rlinf_env.close()

    def get_dataset(self) -> dict:
        """Return the offline D4RL dataset for offline RL training."""
        if hasattr(self._rlinf_env, "env") and hasattr(self._rlinf_env.env, "get_dataset"):
            return self._rlinf_env.env.get_dataset()
        try:
            import d4rl
            import gym
            env = gym.make(self.cfg["task_name"])
            return env.get_dataset()
        except Exception:
            raise RuntimeError(
                f"Cannot load D4RL dataset for {self.cfg['task_name']}. "
                "Ensure d4rl is installed."
            )
