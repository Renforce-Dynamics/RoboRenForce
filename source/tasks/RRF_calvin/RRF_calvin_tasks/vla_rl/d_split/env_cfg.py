"""Env cfg for CALVIN D-split (long-horizon language-conditioned manipulation)."""

from __future__ import annotations

from dataclasses import dataclass

from RRF_calvin_tasks.envs.calvin_env import CalvinRRFEnv


@dataclass
class CalvinDSplitEnvCfg:
    dataset_path: str = ""  # path to CALVIN data dir; required at run time
    num_envs: int = 4
    device: str = "cuda:0"
    image_size: tuple[int, int] = (224, 224)
    action_dim: int = 7
    state_dim: int = 7
    max_episode_steps: int = 360
    reward_coef: float = 1.0

    def build(self) -> CalvinRRFEnv:
        env_dict = {
            "dataset_path": self.dataset_path,
            "image_size": self.image_size,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "max_episode_steps": self.max_episode_steps,
            "reward_coef": self.reward_coef,
        }
        return CalvinRRFEnv(env_dict, num_envs=self.num_envs, device=self.device)
