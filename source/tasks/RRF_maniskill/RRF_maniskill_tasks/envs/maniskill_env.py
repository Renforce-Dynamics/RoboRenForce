"""
ManiSkill Environment Wrapper for RoboRenForce

Directly wraps ManiSkill (SAPIEN-based) behind the EmbodiedEnv interface.
No dependency on RLinf — uses upstream mani_skill package directly.

ManiSkill provides GPU-accelerated manipulation tasks with diverse objects,
camera views, and language instructions.

Observation format:
    main_images:      [B, H, W, 3] - camera RGB (uint8)
    states:           [B, state_dim] - flattened proprioception
    task_descriptions: list[str]     - language instructions

Action format: [B, action_dim] — continuous robot control (varies by task)

Dependencies:
    pip install mani_skill
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from RoboRenForce.prototype.embodied import EmbodiedEnv


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
    seed: int = 0
    reward_coef: float = 1.0


class ManiSkillRRFEnv(EmbodiedEnv):
    """ManiSkill environment implementing EmbodiedEnv.

    Directly uses upstream mani_skill package (no RLinf dependency).
    Supports GPU-accelerated parallel simulation via ManiSkill's built-in vectorization.

    Usage:
        cfg = {
            "task_name": "PickCube-v1",
            "image_size": (224, 224),
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

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.num_actions = cfg.get("action_dim", 7)
        self.num_obs = cfg.get("state_dim", 25)
        self.image_size = cfg.get("image_size", (224, 224))
        self.state_dim = self.num_obs
        self.has_wrist_camera = cfg.get("has_wrist_camera", False)
        self.max_episode_length = cfg.get("max_episode_steps", 200)
        self.num_privileged_obs = 0
        self._reward_coef = cfg.get("reward_coef", 1.0)

        # Buffers
        self.rew_buf = torch.zeros(num_envs, device=self.device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.obs_buf = None
        self.privileged_obs_buf = None
        self.extras = {}

        self._env = None
        self._init_sim(cfg, num_envs)

    def _init_sim(self, cfg: dict, num_envs: int):
        """Create ManiSkill vectorized environment."""
        try:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401  — registers envs
        except ImportError:
            raise ImportError(
                "ManiSkill not found. Install it:\n"
                "  pip install mani_skill"
            )

        self._env = gym.make(
            cfg["task_name"],
            num_envs=num_envs,
            obs_mode=cfg.get("obs_mode", "rgbd"),
            control_mode=cfg.get("control_mode", "pd_ee_delta_pose"),
            reward_mode=cfg.get("reward_mode", "dense"),
            max_episode_steps=cfg.get("max_episode_steps", 200),
        )

    def _wrap_obs(self, raw_obs, info=None) -> dict[str, Any]:
        """Convert ManiSkill observation to EmbodiedEnv format."""
        # ManiSkill returns dict obs with sensor_data and agent keys
        if isinstance(raw_obs, dict) and "sensor_data" in raw_obs:
            sensor = raw_obs["sensor_data"]
            # Try common camera names
            for cam_name in ["3rd_view_camera", "base_camera", "hand_camera"]:
                if cam_name in sensor and "rgb" in sensor[cam_name]:
                    main_images = sensor[cam_name]["rgb"]
                    if isinstance(main_images, torch.Tensor):
                        main_images = main_images.to(self.device, dtype=torch.uint8)
                    break
            else:
                # Fallback: use first available camera
                first_cam = next(iter(sensor.values()))
                main_images = first_cam.get("rgb", torch.zeros(
                    self.num_envs, *self.image_size, 3, dtype=torch.uint8, device=self.device
                ))

            # Proprioception
            try:
                qpos = self._env.unwrapped.agent.robot.get_qpos()
                states = qpos.to(self.device, dtype=torch.float32)
            except Exception:
                states = torch.zeros(self.num_envs, self.state_dim, device=self.device)

            # Language instruction
            try:
                desc = self._env.unwrapped.get_language_instruction()
                if isinstance(desc, str):
                    task_descriptions = [desc] * self.num_envs
                else:
                    task_descriptions = list(desc)
            except Exception:
                task_descriptions = [self.cfg["task_name"]] * self.num_envs

        elif isinstance(raw_obs, torch.Tensor):
            # State-only mode
            return {
                "main_images": torch.zeros(
                    self.num_envs, *self.image_size, 3, dtype=torch.uint8, device=self.device
                ),
                "states": raw_obs.to(self.device, dtype=torch.float32),
                "task_descriptions": [self.cfg["task_name"]] * self.num_envs,
            }
        else:
            raise ValueError(f"Unexpected obs type: {type(raw_obs)}")

        return {
            "main_images": main_images,
            "states": states,
            "task_descriptions": task_descriptions,
        }

    # ---- EmbodiedEnv interface ----

    def get_observations(self) -> tuple[dict[str, Any], dict]:
        raw_obs, info = self._env.unwrapped.get_obs()
        return self._wrap_obs(raw_obs, info), self.extras

    def reset(self) -> tuple[dict[str, Any], dict]:
        raw_obs, info = self._env.reset(seed=self.cfg.get("seed"))
        obs = self._wrap_obs(raw_obs, info)
        self.episode_length_buf.zero_()
        self.reset_buf.zero_()
        self.extras = {"infos": info}
        return obs, self.extras

    def step(self, actions: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict]:
        # ManiSkill accepts torch tensors directly (GPU-accelerated)
        raw_obs, rewards, terminated, truncated, info = self._env.step(actions)
        obs = self._wrap_obs(raw_obs, info)

        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        else:
            rewards = rewards.to(self.device, dtype=torch.float32)
        rewards = rewards * self._reward_coef

        if not isinstance(terminated, torch.Tensor):
            terminated = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated, dtype=torch.bool, device=self.device)

        dones = terminated | truncated
        self.episode_length_buf += 1
        self.rew_buf = rewards
        self.reset_buf = dones
        self.extras = {"termination": terminated, "timeout": truncated, "infos": info}
        return obs, rewards, dones, self.extras

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
