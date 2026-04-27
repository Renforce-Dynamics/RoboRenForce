"""Env cfg for RoboTwin ``place_empty_cup`` (bimanual, 14-dim action).

The cfg is a small dataclass with a ``build()`` method so the script can do:

    env_cfg = spec.kwargs["env_cfg_entry_point"]
    env_cfg.num_envs = args.num_envs           # CLI override
    env = env_cfg.build()
"""

from __future__ import annotations

from dataclasses import dataclass, field

from RRF_robotwin_tasks.envs.robotwin_env import RoboTwinEnv, RoboTwinTaskConfig


@dataclass
class RoboTwinPlaceCupEnvCfg:
    task_name: str = "place_empty_cup"
    num_envs: int = 8
    device: str = "cuda:0"
    image_size: tuple[int, int] = (224, 224)
    action_dim: int = 14
    state_dim: int = 14
    max_episode_steps: int = 200
    reward_coef: float = 5.0
    embodiment: list = field(default_factory=lambda: ["piper", "piper", 0.6])

    def build(self) -> RoboTwinEnv:
        task_cfg = RoboTwinTaskConfig(
            task_name=self.task_name,
            planner_backend="mplib",
            embodiment=self.embodiment,
            step_lim=self.max_episode_steps,
        )
        env_dict = {
            "task_config": task_cfg,
            "image_size": self.image_size,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "max_episode_steps": self.max_episode_steps,
            "use_custom_reward": True,
            "use_rel_reward": True,
            "reward_coef": self.reward_coef,
            "center_crop": True,
        }
        return RoboTwinEnv(env_dict, num_envs=self.num_envs, device=self.device)
