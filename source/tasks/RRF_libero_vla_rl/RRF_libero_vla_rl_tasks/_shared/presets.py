"""LIBERO actor + algo presets (single-arm, 7-dim action)."""

from __future__ import annotations

from RoboRenForce.networks.vlm.qwen2vl import Qwen2VLCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import (
    RegressionActionHeadCfg,
)
from RoboRenForce.components.actor.vla_actor import VLAActorCfg
from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithmCfg


def libero_vla_actor(action_dim: int = 7) -> VLAActorCfg:
    return VLAActorCfg(
        vlm_backbone_cfg=Qwen2VLCfg(
            model_name="Qwen/Qwen2-VL-2B-Instruct", freeze=True, device_map_auto=False,
        ),
        freeze_vlm=True,
        fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
        action_head_cfg=RegressionActionHeadCfg(
            action_dim=action_dim, action_horizon=1, hidden_dims=[256, 256],
        ),
        use_proprioception=True,
        use_text=True,
    )


def libero_grpo_default() -> GRPOAlgorithmCfg:
    return GRPOAlgorithmCfg(
        group_size=8, clip_ratio_low=0.2, clip_ratio_high=0.28,
        kl_beta=0.05, update_epochs=4, learning_rate=1e-4, reward_coef=1.0,
    )
