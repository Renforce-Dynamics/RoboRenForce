"""
VLA Actor (System 1 + System 2)

Combines VLM backbone (System 2) with Action Expert (System 1).

Reference: .references/Psi0/src/psi/models/psi0.py

TODO Phase 2 (Week 2, Priority P0):
- [ ] Implement VLA actor combining VLM + fusion + action head
- [ ] Support frozen VLM backbone
- [ ] Handle text and proprioception inputs
- [ ] Implement forward pass
- [ ] Add LoRA support for VLM (optional)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass, MISSING
from RoboRenForce.components.actor.actor_base import ActorBase, ActorBaseCfg
from RoboRenForce.networks.vlm import VLMBackboneCfg, FusionLayerCfg
from RoboRenForce.components.actor.action_heads import DiffusionActionHeadCfg


@configclass
class VLAActorCfg(ActorBaseCfg):
    """
    VLA Actor configuration.

    Architecture:
    1. System 2: VLM backbone (frozen) extracts VL features
    2. Fusion: Combines VL features + proprioception
    3. System 1: Action Expert predicts actions

    Training strategy:
    - Pretrain: Train fusion + action head, freeze VLM
    - SFT: Fine-tune fusion + action head on task data
    - RL Fine-tune: RL optimization with LoRA on action head
    """

    class_type: type["VLAActor"] = MISSING

    # System 2: VLM backbone
    vlm_backbone_cfg: VLMBackboneCfg = MISSING
    freeze_vlm: bool = True

    # Fusion layer
    fusion_cfg: FusionLayerCfg = FusionLayerCfg()

    # System 1: Action Expert
    action_head_cfg: DiffusionActionHeadCfg = MISSING  # or RegressionActionHeadCfg

    # Input modalities
    use_proprioception: bool = True
    use_text: bool = True


class VLAActor(ActorBase):
    """
    VLA Actor implementation.

    Forward pass:
    1. VLM extracts VL features (frozen System 2)
    2. Fusion combines VL + proprioception
    3. Action head predicts actions (trainable System 1)

    TODO Phase 2:
    - [ ] Construct VLM backbone from config
    - [ ] Construct fusion layer
    - [ ] Construct action head
    - [ ] Implement forward pass
    - [ ] Handle freezing of VLM
    - [ ] Support optional text input
    """

    def __init__(self, cfg: VLAActorCfg, dim_params: dict):
        super().__init__(cfg, dim_params)

        # TODO: System 2 - VLM backbone
        # self.vlm = cfg.vlm_backbone_cfg.construct_from_cfg()
        # if cfg.freeze_vlm:
        #     for param in self.vlm.parameters():
        #         param.requires_grad = False
        raise NotImplementedError("TODO: Construct VLM backbone")

        # TODO: Fusion layer
        # fusion_dim_params = {
        #     "vl_feature_dim": self.vlm.output_dim,
        #     "proprio_dim": dim_params.get("proprioception_dim", 0),
        # }
        # self.fusion = cfg.fusion_cfg.construct_from_cfg(fusion_dim_params)
        raise NotImplementedError("TODO: Construct fusion layer")

        # TODO: System 1 - Action Expert
        # action_dim_params = {
        #     "input_dim": self.fusion.output_dim,
        #     "action_dim": dim_params["action_dim"],
        # }
        # self.action_head = cfg.action_head_cfg.construct_from_cfg(action_dim_params)
        raise NotImplementedError("TODO: Construct action head")

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through VLA.

        Args:
            obs_dict: {
                "image": (B, C, H, W),
                "text": (B, max_text_len) or None,
                "proprioception": (B, proprio_dim) or None,
            }
            deterministic: If True, use deterministic sampling (for eval)

        Returns:
            actions: (B, action_dim) or (B, action_horizon, action_dim)

        TODO:
        - Extract VL features from VLM (with grad enabled/disabled based on freeze)
        - Fuse VL features with proprioception
        - Predict actions with action head
        - Return actions
        """
        raise NotImplementedError("TODO: Implement VLA forward pass")

        # Example structure:
        # # System 2: VLM features (frozen or not)
        # with torch.set_grad_enabled(not self.cfg.freeze_vlm):
        #     vl_features = self.vlm(
        #         image=obs_dict["image"],
        #         text=obs_dict.get("text", None) if self.cfg.use_text else None,
        #     )
        #
        # # Fusion
        # if self.cfg.use_proprioception:
        #     fused_features = self.fusion(vl_features, obs_dict["proprioception"])
        # else:
        #     fused_features = vl_features
        #
        # # System 1: Action prediction
        # actions = self.action_head(fused_features, deterministic=deterministic)
        #
        # return actions

    def get_action(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get action for deployment (wrapper around forward).

        TODO:
        - Call forward with deterministic=True
        - Return actions
        """
        return self.forward(obs_dict, deterministic=True)
