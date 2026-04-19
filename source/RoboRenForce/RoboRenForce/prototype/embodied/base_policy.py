"""
BasePolicy — Unified interface for all embodied policies.

A "policy" here = everything needed to go from multimodal obs → actions.
This could be a VLM + action head, a diffusion policy, or a simple MLP.

The same policy supports multiple training modes via ForwardType dispatch:
- PRETRAIN: supervised action prediction (L1/L2/diffusion loss)
- SFT: supervised fine-tune with KL regularization
- PPO: on-policy RL (returns logprobs, values, entropy)
- SAC: off-policy RL (returns Q-values)
- INFERENCE: pure action prediction (no gradients)

Concrete implementations live in RRF_models package (separate from core).

Reference: RLinf rlinf/models/embodiment/base_policy.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn


class ForwardType(Enum):
    """Training mode selector for policy forward pass."""
    INFERENCE = "inference"     # Pure action prediction (no loss)
    PRETRAIN = "pretrain"       # Supervised pretraining (action loss only)
    SFT = "sft"                 # Supervised fine-tune (action loss + KL)
    PPO = "ppo"                 # On-policy RL (logprobs + values + entropy)
    SAC = "sac"                 # Off-policy RL (Q-values)


class BasePolicy(ABC, nn.Module):
    """Abstract base class for all embodied policies.

    Subclass in RRF_models to wrap specific pretrained models (Qwen2-VL,
    OpenPI, GR00T, etc.) behind this unified interface.

    Observation format (standardized by EmbodiedEnv):
        obs["main_images"]:       [B, H, W, C]  primary camera
        obs["wrist_images"]:      [B, H, W, C]  wrist camera (optional)
        obs["states"]:            [B, state_dim] proprioception
        obs["task_descriptions"]: list[str]      language instructions

    Action format:
        actions: [B, action_horizon, action_dim]
    """

    def __init__(self):
        super().__init__()

    # ---- Core interface ----

    @abstractmethod
    def predict_action(
        self,
        obs: dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        """Predict actions from observations (inference mode, no loss).

        Args:
            obs: standardized observation dict from EmbodiedEnv

        Returns:
            actions: [B, action_horizon, action_dim]
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        forward_type: ForwardType = ForwardType.PRETRAIN,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass. Dispatch to mode-specific logic.

        Args:
            forward_type: which training mode to use
            **kwargs: mode-specific inputs (obs, actions, targets, etc.)

        Returns:
            dict with at least {"loss": scalar} and mode-specific extras:
            - PRETRAIN: {"loss", "action_loss"}
            - SFT:      {"loss", "action_loss", "kl_loss"}
            - PPO:      {"loss", "logprobs", "entropy", "values"}
            - SAC:      {"loss", "q_values"}
        """
        raise NotImplementedError

    # ---- Optional overrides ----

    def pretrain_forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Supervised pretraining: obs + gt_actions → action loss."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement pretrain_forward"
        )

    def sft_forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Supervised fine-tune: action loss + KL to reference."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement sft_forward"
        )

    def ppo_forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """PPO: compute logprobs, values, entropy for collected rollout."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement ppo_forward"
        )

    def sac_forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """SAC: compute Q-values for off-policy update."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement sac_forward"
        )

    # ---- Backbone management ----

    def freeze_backbone(self):
        """Freeze the vision-language backbone (keep action head trainable)."""
        pass  # Override in subclass

    def unfreeze_backbone(self):
        """Unfreeze the backbone for full fine-tuning."""
        pass  # Override in subclass

    def get_value(self, obs: dict[str, Any]) -> Optional[torch.Tensor]:
        """Compute value estimate (for RL with value head). Returns None if no value head."""
        return None

    # ---- Utility ----

    @property
    def has_value_head(self) -> bool:
        return hasattr(self, "value_head") and self.value_head is not None

    def trainable_parameters(self):
        """Return only parameters that require grad (for optimizer)."""
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
