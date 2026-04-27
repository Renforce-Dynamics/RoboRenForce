"""Env cfg for ManiSkill PickCube-v1 (7-dim Franka, dense reward)."""

from __future__ import annotations

from dataclasses import dataclass

from RRF_maniskill_tasks.envs.maniskill_env import ManiSkillRRFEnv


@dataclass
class ManiSkillPickCubeEnvCfg:
    task_name: str = "PickCube-v1"
    num_envs: int = 8
    device: str = "cuda:0"
    image_size: tuple[int, int] = (224, 224)
    action_dim: int = 7
    state_dim: int = 25
    max_episode_steps: int = 200
    reward_coef: float = 1.0
    obs_mode: str = "rgbd"
    control_mode: str = "pd_ee_delta_pose"
    reward_mode: str = "dense"

    def build(self) -> ManiSkillRRFEnv:
        env_dict = {
            "task_name": self.task_name,
            "image_size": self.image_size,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "max_episode_steps": self.max_episode_steps,
            "reward_coef": self.reward_coef,
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "reward_mode": self.reward_mode,
        }
        return ManiSkillRRFEnv(env_dict, num_envs=self.num_envs, device=self.device)
