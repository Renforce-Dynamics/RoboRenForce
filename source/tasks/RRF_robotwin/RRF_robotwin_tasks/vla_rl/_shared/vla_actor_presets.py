"""Reusable VLA actor presets for RoboTwin tasks.

Tasks combine these presets with task-specific overrides
(action_dim, image size, etc.) inside their ``actor.py`` module.
"""

from __future__ import annotations

from RoboRenForce.networks.vlm.qwen2vl import Qwen2VLCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import (
    RegressionActionHeadCfg,
)
from RoboRenForce.components.actor.vla_actor import VLAActorCfg


def qwen2vl_2b_backbone(freeze: bool = True) -> Qwen2VLCfg:
    """Default Qwen2-VL-2B backbone preset.

    ``freeze=True`` keeps VLM weights frozen and trains only fusion + action head
    (the recommended starting point for RL fine-tuning).
    """
    return Qwen2VLCfg(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        freeze=freeze,
        device_map_auto=False,
    )


def regression_head(action_dim: int, action_horizon: int = 1) -> RegressionActionHeadCfg:
    return RegressionActionHeadCfg(
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dims=[256, 256],
    )


def qwen2vl_actor(
    action_dim: int,
    action_horizon: int = 1,
    freeze_vlm: bool = True,
) -> VLAActorCfg:
    """Compose a full VLAActorCfg with Qwen2-VL backbone + regression head."""
    return VLAActorCfg(
        vlm_backbone_cfg=qwen2vl_2b_backbone(freeze=freeze_vlm),
        freeze_vlm=freeze_vlm,
        fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
        action_head_cfg=regression_head(action_dim, action_horizon),
        use_proprioception=True,
        use_text=True,
    )
