"""Env cfg for LIBERO spatial pick task (7-dim Franka)."""

from __future__ import annotations

from dataclasses import dataclass

from RRF_libero_tasks.envs.libero_env import LiberoRRFEnv


@dataclass
class LiberoSpatialEnvCfg:
    task_suite_name: str = "libero_spatial"
    num_envs: int = 8
    device: str = "cuda:0"
    image_size: tuple[int, int] = (224, 224)
    action_dim: int = 7
    state_dim: int = 7
    max_episode_steps: int = 300
    reward_coef: float = 1.0

    def build(self) -> LiberoRRFEnv:
        env_dict = {
            "task_suite_name": self.task_suite_name,
            "image_size": self.image_size,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "max_episode_steps": self.max_episode_steps,
            "reward_coef": self.reward_coef,
            "has_wrist_camera": True,
        }
        return LiberoRRFEnv(env_dict, num_envs=self.num_envs, device=self.device)
