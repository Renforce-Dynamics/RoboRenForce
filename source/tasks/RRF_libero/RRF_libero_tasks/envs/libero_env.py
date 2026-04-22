"""
LIBERO Environment Wrapper for RoboRenForce

Wraps RLinf's LiberoEnv behind the EmbodiedEnv interface.

LIBERO task suites:
    - libero_10:       10 manipulation tasks (standard benchmark)
    - libero_spatial:  90 tasks testing spatial reasoning
    - libero_object:   90 tasks testing object generalization
    - libero_goal:     90 tasks testing goal-driven behavior
    - libero_90:       90-task aggregate
    - libero_130:      all 130 tasks combined

Observation format (from RLinf):
    main_images:      [B, H, W, 3] - front camera RGB
    wrist_images:     [B, H, W, 3] - wrist camera RGB
    states:           [B, 7]       - EEF pos(3) + axisangle(3) + gripper(1)
    task_descriptions: list[str]   - language instructions

Action format: [B, 7] - 6D EEF pose + 1D gripper
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from RoboRenForce.utils.env_wrapper.rlinf_bridge import RLinfBridgeEnv


@dataclass
class LiberoTaskConfig:
    """Configuration for LIBERO environment."""
    task_suite_name: str = "libero_10"
    image_size: tuple[int, int] = (224, 224)
    action_dim: int = 7
    state_dim: int = 7
    max_episode_steps: int = 300
    has_wrist_camera: bool = True

    # RLinf-specific
    seed: int = 0
    group_size: int = 1
    use_fixed_reset_state_ids: bool = False
    ignore_terminations: bool = False
    auto_reset: bool = True
    use_rel_reward: bool = False
    use_step_penalty: bool = False
    reward_coef: float = 1.0

    # Init params passed to OffScreenRenderEnv
    init_params: dict = field(default_factory=lambda: {
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "render_camera": "agentview",
        "ignore_done": True,
        "use_camera_obs": True,
        "camera_depths": False,
        "camera_heights": 128,
        "camera_widths": 128,
        "reward_shaping": True,
    })

    # Video recording
    video_cfg: dict = field(default_factory=lambda: {"enabled": False})


class LiberoRRFEnv(RLinfBridgeEnv):
    """LIBERO environment implementing EmbodiedEnv via RLinf bridge.

    Usage:
        cfg = {
            "task_suite_name": "libero_10",
            "image_size": (224, 224),
            "action_dim": 7,
            "state_dim": 7,
            "max_episode_steps": 300,
            "has_wrist_camera": True,
        }
        env = LiberoRRFEnv(cfg, num_envs=8, device="cuda:0")
        obs, info = env.reset()
        obs, rewards, dones, info = env.step(actions)
    """

    def __init__(self, cfg: dict, num_envs: int = 1, device: str = "cpu", **kwargs):
        # Apply defaults from LiberoTaskConfig
        defaults = LiberoTaskConfig()
        for k, v in defaults.__dict__.items():
            cfg.setdefault(k, v)

        super().__init__(cfg, num_envs, device, **kwargs)
        self.has_wrist_camera = cfg.get("has_wrist_camera", True)

        # Build RLinf config object (omegaconf-compatible)
        self._init_rlinf_env(cfg, num_envs, kwargs)

    def _init_rlinf_env(self, cfg: dict, num_envs: int, kwargs: dict):
        """Construct the underlying RLinf LiberoEnv."""
        try:
            from omegaconf import OmegaConf
            from rlinf.envs.libero.libero_env import LiberoEnv
        except ImportError:
            raise ImportError(
                "LIBERO environment requires rlinf and libero packages:\n"
                "  pip install rlinf\n"
                "  pip install libero robosuite"
            )

        # Convert dict config to OmegaConf for RLinf compatibility
        rlinf_cfg = OmegaConf.create({
            "task_suite_name": cfg["task_suite_name"],
            "seed": cfg.get("seed", 0),
            "group_size": cfg.get("group_size", 1),
            "use_fixed_reset_state_ids": cfg.get("use_fixed_reset_state_ids", False),
            "ignore_terminations": cfg.get("ignore_terminations", False),
            "auto_reset": cfg.get("auto_reset", True),
            "use_rel_reward": cfg.get("use_rel_reward", False),
            "use_step_penalty": cfg.get("use_step_penalty", False),
            "reward_coef": cfg.get("reward_coef", 1.0),
            "init_params": cfg.get("init_params", LiberoTaskConfig().init_params),
            "video_cfg": cfg.get("video_cfg", {"enabled": False}),
        })

        seed_offset = kwargs.get("seed_offset", 0)
        total_num_processes = kwargs.get("total_num_processes", 1)
        worker_info = kwargs.get("worker_info", None)

        self._rlinf_env = LiberoEnv(
            cfg=rlinf_cfg,
            num_envs=num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=worker_info,
        )
