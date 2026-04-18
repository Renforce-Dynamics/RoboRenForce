"""
VLA Actor (System 1 + System 2)

Combines VLM backbone (System 2) with Fusion + Action Expert (System 1).

IO Contract:
    Input (forward):
        obs_dict: {
            "image":          (B, C, H, W) float tensor,
            "text":           str or list[str] (optional),
            "proprioception": (B, proprio_dim) float tensor (optional),
        }
        deterministic: bool

    Output (forward):
        actions: (B, action_horizon, action_dim)

    Training (train_forward):
        Input:  obs_dict, target_actions (B, action_horizon, action_dim)
        Output: dict from action_head.train_forward (for loss computation)
"""

from __future__ import annotations

from typing import Dict
from dataclasses import MISSING

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.vlm.vlm_backbone_base import VLMBackboneCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg, FusionLayer


class VLAActor(ModuleBase):
    """
    VLA Actor implementation.

    Forward pass:
    1. VLM extracts VL features (frozen System 2)
    2. Fusion combines VL + proprioception
    3. Action head predicts actions (trainable System 1)
    """

    def __init__(self, cfg: VLAActorCfg, dim_params: dict):
        super().__init__()
        self.cfg = cfg

        # System 2: VLM backbone
        self.vlm = cfg.vlm_backbone_cfg.construct_from_cfg()
        if cfg.freeze_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False

        # Fusion layer
        vl_feature_dim = self.vlm.output_dim
        proprio_dim = dim_params.get("proprioception_dim", 0)

        if cfg.use_proprioception and proprio_dim > 0:
            fusion_dim_params = {
                "vl_feature_dim": vl_feature_dim,
                "proprio_dim": proprio_dim,
            }
            self.fusion = cfg.fusion_cfg.construct_from_cfg(fusion_dim_params)
            action_input_dim = self.fusion.output_dim
        else:
            self.fusion = None
            action_input_dim = vl_feature_dim

        # System 1: Action Expert
        action_dim_params = {"input_dim": action_input_dim}
        self.action_head = cfg.action_head_cfg.construct_from_cfg(action_dim_params)

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        target_actions: torch.Tensor = None,
        deterministic: bool = False,
    ):
        """
        Unified forward pass for both training and inference.

        When target_actions is provided, returns training output dict (for loss).
        When target_actions is None, returns predicted actions tensor.

        This design ensures DDP properly intercepts all forward computation.
        """
        # System 2: VLM features (frozen or not)
        with torch.set_grad_enabled(not self.cfg.freeze_vlm):
            vl_features = self.vlm(
                image=obs_dict["image"],
                text=obs_dict.get("text", None) if self.cfg.use_text else None,
            )

        # Fusion
        if self.fusion is not None and "proprioception" in obs_dict:
            fused_features = self.fusion(vl_features, obs_dict["proprioception"])
        else:
            fused_features = vl_features

        # System 1: Action prediction
        if target_actions is not None:
            return self.action_head.train_forward(fused_features, target_actions)
        return self.action_head(fused_features, deterministic=deterministic)

    def train_forward(self, obs_dict: Dict[str, torch.Tensor], target_actions: torch.Tensor) -> dict:
        """Convenience wrapper — calls forward() with target_actions."""
        return self.forward(obs_dict, target_actions=target_actions)

    def get_action(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(obs_dict, deterministic=True)


@configclass
class VLAActorCfg(ModuleBaseCfg):
    """VLA Actor configuration."""

    class_type: type[VLAActor] = VLAActor

    vlm_backbone_cfg: VLMBackboneCfg = MISSING
    freeze_vlm: bool = True
    fusion_cfg: FusionLayerCfg = FusionLayerCfg()
    action_head_cfg: ModuleBaseCfg = MISSING
    use_proprioception: bool = True
    use_text: bool = True
