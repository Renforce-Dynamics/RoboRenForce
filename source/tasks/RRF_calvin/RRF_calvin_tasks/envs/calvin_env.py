"""
CALVIN Environment Wrapper for RoboRenForce

Wraps RLinf's CalvinEnv behind the EmbodiedEnv interface.

CALVIN evaluates long-horizon manipulation via 5-subtask sequences.
Each subtask is a natural language instruction that the agent must complete
before moving to the next subtask.

Task suites:
    - calvin_d:    scene D only
    - calvin_abc:  scenes A, B, C
    - calvin_abcd: all four scenes

Observation format (from RLinf):
    main_images:      [B, H, W, 3] - static camera RGB
    wrist_images:     [B, H, W, 3] - gripper camera RGB
    states:           [B, 7]       - 7-DOF joint positions
    task_descriptions: list[str]   - current subtask language instruction

Action format: [B, 7] - 6D EEF pose + 1D gripper
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from RoboRenForce.utils.env_wrapper.rlinf_bridge import RLinfBridgeEnv


@dataclass
class CalvinTaskConfig:
    """Configuration for CALVIN environment."""
    task_suite_name: str = "calvin_abcd"
    image_size: tuple[int, int] = (224, 224)
    action_dim: int = 7
    state_dim: int = 7
    max_episode_steps: int = 360   # CALVIN uses 360 steps per subtask
    has_wrist_camera: bool = True
    num_subtasks: int = 5

    # RLinf-specific
    seed: int = 0
    auto_reset: bool = True
    use_rel_reward: bool = False
    reward_coef: float = 1.0

    # CALVIN data paths
    dataset_path: str = ""
    calvin_env_cfg: dict = field(default_factory=dict)


class CalvinRRFEnv(RLinfBridgeEnv):
    """CALVIN environment implementing EmbodiedEnv via RLinf bridge.

    Usage:
        cfg = {
            "task_suite_name": "calvin_abcd",
            "image_size": (224, 224),
            "action_dim": 7,
            "state_dim": 7,
            "max_episode_steps": 360,
            "has_wrist_camera": True,
            "dataset_path": "/path/to/calvin/dataset",
        }
        env = CalvinRRFEnv(cfg, num_envs=4, device="cuda:0")
        obs, info = env.reset()
        obs, rewards, dones, info = env.step(actions)
    """

    def __init__(self, cfg: dict, num_envs: int = 1, device: str = "cpu", **kwargs):
        defaults = CalvinTaskConfig()
        for k, v in defaults.__dict__.items():
            cfg.setdefault(k, v)

        super().__init__(cfg, num_envs, device, **kwargs)
        self.has_wrist_camera = True
        self._init_rlinf_env(cfg, num_envs, kwargs)

    def _init_rlinf_env(self, cfg: dict, num_envs: int, kwargs: dict):
        """Construct the underlying RLinf CalvinEnv."""
        try:
            from omegaconf import OmegaConf
            from rlinf.envs.calvin.calvin_gym_env import CalvinEnv
        except ImportError:
            raise ImportError(
                "CALVIN environment requires rlinf and calvin_env:\n"
                "  pip install rlinf\n"
                "  # Follow CALVIN setup: https://github.com/mees/calvin"
            )

        rlinf_cfg = OmegaConf.create({
            "task_suite_name": cfg["task_suite_name"],
            "seed": cfg.get("seed", 0),
            "auto_reset": cfg.get("auto_reset", True),
            "use_rel_reward": cfg.get("use_rel_reward", False),
            "reward_coef": cfg.get("reward_coef", 1.0),
            "dataset_path": cfg.get("dataset_path", ""),
            "calvin_env_cfg": cfg.get("calvin_env_cfg", {}),
        })

        seed_offset = kwargs.get("seed_offset", 0)
        total_num_processes = kwargs.get("total_num_processes", 1)
        worker_info = kwargs.get("worker_info", None)

        self._rlinf_env = CalvinEnv(
            cfg=rlinf_cfg,
            num_envs=num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=worker_info,
        )
