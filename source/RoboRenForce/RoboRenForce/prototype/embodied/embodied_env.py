"""
EmbodiedEnv — Vectorized environment with multimodal observations.

Extends the classic VecEnv (state-vector obs) to support dict-based
observations containing images, language instructions, and proprioception.

Both VLA pretraining (offline dataset → EmbodiedOutput) and VLA RL
(sim interaction → EmbodiedOutput) share the same observation interface.

Concrete implementations live in task packages:
- source/tasks/RRF_robotwin/envs/robotwin_env.py
- source/tasks/RRF_isaaclab/embodied_wrapper/...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from RoboRenForce.prototype.classic.vec_env import RoboRenForceVecEnv


class EmbodiedEnv(RoboRenForceVecEnv):
    """Vectorized environment with multimodal (image + text + proprio) observations.

    Observation dict keys (standardized across all embodied envs):
        main_images:   [B, H, W, C]  — primary camera view (RGB, uint8 or float)
        wrist_images:  [B, H, W, C]  — wrist/egocentric camera (optional)
        states:        [B, state_dim] — proprioception (joint pos/vel, etc.)
        task_descriptions: list[str]  — natural language instructions (len = B)

    Subclass this in each benchmark's task package and implement
    step() / reset() / get_observations().
    """

    # Additional metadata beyond classic VecEnv
    image_size: tuple[int, int] = (224, 224)
    state_dim: int = 0
    has_wrist_camera: bool = False

    @abstractmethod
    def get_observations(self) -> tuple[dict[str, Any], dict]:
        """Return multimodal observation dict + extras.

        Returns:
            obs_dict: {main_images, wrist_images?, states, task_descriptions}
            extras: additional info (episode metrics, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> tuple[dict[str, Any], dict]:
        """Reset all envs, return multimodal obs dict + extras."""
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict]:
        """Step all envs.

        Args:
            actions: [B, action_dim]

        Returns:
            obs_dict: {main_images, wrist_images?, states, task_descriptions}
            rewards:  [B]
            dones:    [B]
            extras:   dict with at least {timeout, termination}
        """
        raise NotImplementedError
