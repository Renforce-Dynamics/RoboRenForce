"""
Multimodal Environment Wrapper

Wraps a state-vector VecEnv (e.g. Isaac Lab) to provide multimodal observations
compatible with the EmbodiedEnv interface: images, proprioception, task descriptions.

This bridges the gap between classic RL envs (state obs only) and VLA policies
that need camera images + language instructions.

Usage:
    base_env = IsaacLabEnv(...)  # Classic state-vector env
    vla_env = MultimodalEnvWrapper(base_env, cfg)
    obs, info = vla_env.reset()
    # obs = {main_images: [B,H,W,3], states: [B,D], task_descriptions: [...]}
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBaseCfg
from RoboRenForce.prototype.embodied.embodied_env import EmbodiedEnv


class MultimodalEnvWrapper(EmbodiedEnv):
    """Wraps a state-vector env to provide multimodal obs for VLA policies.

    Supports two image modes:
    - "render": render from simulator cameras (if available)
    - "dummy": generate dummy images (for testing without rendering)

    The wrapper always provides: main_images, states, task_descriptions.
    Optionally provides: wrist_images (if has_wrist_camera=True).
    """

    def __init__(self, base_env, cfg: "MultimodalEnvWrapperCfg"):
        self._base_env = base_env
        self._cfg = cfg

        # Inherit base env properties
        self.num_envs = base_env.num_envs
        self.num_actions = base_env.num_actions
        self.num_obs = base_env.num_obs
        self.max_episode_length = getattr(base_env, "max_episode_length", cfg.max_episode_length)
        self.device = getattr(base_env, "device", torch.device("cpu"))

        # EmbodiedEnv metadata
        self.image_size = cfg.image_size
        self.state_dim = base_env.num_obs
        self.has_wrist_camera = cfg.has_wrist_camera

        self._task_description = cfg.task_description
        self._image_mode = cfg.image_mode

    def get_observations(self) -> tuple[dict[str, Any], dict]:
        base_obs, extras = self._base_env.get_observations()
        return self._wrap_obs(base_obs, extras), extras

    def reset(self) -> tuple[dict[str, Any], dict]:
        result = self._base_env.reset()
        if isinstance(result, tuple):
            base_obs, info = result
        else:
            base_obs = result
            info = {}
        return self._wrap_obs(base_obs, info), info

    def step(self, actions: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict]:
        result = self._base_env.step(actions)
        if len(result) == 5:
            # IsaacLab style: obs, rewards, terminated, truncated, info
            base_obs, rewards, terminated, truncated, info = result
            dones = terminated | truncated
            extras = {"termination": terminated, "timeout": truncated, **info}
        elif len(result) == 4:
            base_obs, rewards, dones, extras = result
        else:
            raise ValueError(f"Unexpected step() return length: {len(result)}")

        return self._wrap_obs(base_obs, extras), rewards, dones, extras

    def _wrap_obs(self, base_obs, extras: dict = None) -> dict[str, Any]:
        """Convert base env obs to multimodal obs dict."""
        # Extract state tensor
        if isinstance(base_obs, dict):
            states = base_obs.get("obs", base_obs.get("policy", base_obs.get("states")))
            if states is None:
                # Try concatenating all tensor values
                tensors = [v for v in base_obs.values() if isinstance(v, torch.Tensor)]
                states = torch.cat(tensors, dim=-1) if tensors else torch.zeros(
                    self.num_envs, self.state_dim, device=self.device)
        elif isinstance(base_obs, torch.Tensor):
            states = base_obs
        else:
            states = torch.zeros(self.num_envs, self.state_dim, device=self.device)

        # Get images
        images = self._get_images(extras)

        obs = {
            "main_images": images,
            "states": states,
            "task_descriptions": [self._task_description] * self.num_envs,
        }

        if self.has_wrist_camera:
            obs["wrist_images"] = self._get_wrist_images(extras)

        return obs

    def _get_images(self, extras: dict = None) -> torch.Tensor:
        """Get camera images from simulator or generate dummy ones."""
        H, W = self.image_size

        if self._image_mode == "render" and hasattr(self._base_env, "render"):
            images = self._base_env.render()
            if images is not None:
                if images.shape[1:3] != (H, W):
                    # Resize: [B, H', W', C] -> [B, H, W, C]
                    images = images.permute(0, 3, 1, 2).float()
                    images = F.interpolate(images, size=(H, W), mode="bilinear")
                    images = images.permute(0, 2, 3, 1).to(torch.uint8)
                return images

        if self._image_mode == "render" and extras and "images" in extras:
            return extras["images"]

        # Dummy mode: generate random images
        return torch.randint(
            0, 255, (self.num_envs, H, W, 3),
            dtype=torch.uint8, device=self.device,
        )

    def _get_wrist_images(self, extras: dict = None) -> torch.Tensor:
        """Get wrist camera images."""
        H, W = self.image_size

        if extras and "wrist_images" in extras:
            return extras["wrist_images"]

        return torch.randint(
            0, 255, (self.num_envs, H, W, 3),
            dtype=torch.uint8, device=self.device,
        )

    def close(self):
        if hasattr(self._base_env, "close"):
            self._base_env.close()


@configclass
class MultimodalEnvWrapperCfg(ModuleBaseCfg):
    """Multimodal env wrapper configuration."""

    class_type: type[MultimodalEnvWrapper] = MultimodalEnvWrapper

    image_size: tuple = (224, 224)
    image_mode: str = "dummy"           # "render" or "dummy"
    has_wrist_camera: bool = False
    task_description: str = "complete the task"
    max_episode_length: int = 200
