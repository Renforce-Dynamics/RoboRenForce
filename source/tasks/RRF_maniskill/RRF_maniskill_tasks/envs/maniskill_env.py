"""
ManiSkill Environment Wrapper for RoboRenForce

Wraps RLinf's ManiskillEnv behind the EmbodiedEnv interface.

ManiSkill provides GPU-accelerated manipulation tasks via SAPIEN simulator
with diverse objects, camera views, and language instructions.

Observation format (from RLinf):
    main_images:       [B, H, W, 3] - 3rd-view camera RGB (uint8)
    extra_view_images: [B, V, H, W, 3] - multi-camera views (optional)
    states:            [B, state_dim] - flattened proprioception
    task_descriptions: list[str]     - language instructions

Action format: [B, action_dim] - continuous robot control (varies by task)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from RoboRenForce.utils.env_wrapper.rlinf_bridge import RLinfBridgeEnv


@dataclass
class ManiSkillTaskConfig:
    """Configuration for ManiSkill environment."""
    task_name: str = "PickCube-v1"
    image_size: tuple[int, int] = (224, 224)
    action_dim: int = 7
    state_dim: int = 25
    max_episode_steps: int = 200
    has_wrist_camera: bool = False
    obs_mode: str = "rgbd"
    control_mode: str = "pd_ee_delta_pose"
    reward_mode: str = "dense"

    # RLinf-specific
    seed: int = 0
    wrap_obs_mode: str = "default"
    use_rel_reward: bool = False
    reward_coef: float = 1.0
    auto_reset: bool = True
    use_step_penalty: bool = False


class ManiSkillRRFEnv(RLinfBridgeEnv):
    """ManiSkill environment implementing EmbodiedEnv via RLinf bridge.

    Usage:
        cfg = {
            "task_name": "PickCube-v1",
            "image_size": (224, 224),
            "action_dim": 7,
            "state_dim": 25,
            "max_episode_steps": 200,
            "obs_mode": "rgbd",
            "control_mode": "pd_ee_delta_pose",
        }
        env = ManiSkillRRFEnv(cfg, num_envs=16, device="cuda:0")
        obs, info = env.reset()
        obs, rewards, dones, info = env.step(actions)
    """

    def __init__(self, cfg: dict, num_envs: int = 1, device: str = "cpu", **kwargs):
        defaults = ManiSkillTaskConfig()
        for k, v in defaults.__dict__.items():
            cfg.setdefault(k, v)

        super().__init__(cfg, num_envs, device, **kwargs)
        self._init_rlinf_env(cfg, num_envs, kwargs)

    def _init_rlinf_env(self, cfg: dict, num_envs: int, kwargs: dict):
        """Construct the underlying RLinf ManiskillEnv."""
        try:
            from omegaconf import OmegaConf
            from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
        except ImportError:
            raise ImportError(
                "ManiSkill environment requires rlinf and mani_skill:\n"
                "  pip install rlinf\n"
                "  pip install mani_skill"
            )

        rlinf_cfg = OmegaConf.create({
            "env_args": {
                "env_id": cfg["task_name"],
                "obs_mode": cfg.get("obs_mode", "rgbd"),
                "control_mode": cfg.get("control_mode", "pd_ee_delta_pose"),
                "reward_mode": cfg.get("reward_mode", "dense"),
                "max_episode_steps": cfg.get("max_episode_steps", 200),
            },
            "seed": cfg.get("seed", 0),
            "wrap_obs_mode": cfg.get("wrap_obs_mode", "default"),
            "use_rel_reward": cfg.get("use_rel_reward", False),
            "reward_coef": cfg.get("reward_coef", 1.0),
            "auto_reset": cfg.get("auto_reset", True),
            "use_step_penalty": cfg.get("use_step_penalty", False),
        })

        seed_offset = kwargs.get("seed_offset", 0)
        total_num_processes = kwargs.get("total_num_processes", 1)
        worker_info = kwargs.get("worker_info", None)

        self._rlinf_env = ManiskillEnv(
            cfg=rlinf_cfg,
            num_envs=num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=worker_info,
        )

    def _post_process_obs(self, obs_dict: dict) -> dict:
        """Rename extra_view_images if present (ManiSkill multi-cam)."""
        # ManiSkill may return extra_view_images instead of wrist_images
        if "extra_view_images" in obs_dict and "wrist_images" not in obs_dict:
            # Keep as-is; downstream VLA can handle extra views
            pass
        return obs_dict
