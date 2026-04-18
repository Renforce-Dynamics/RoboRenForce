from dataclasses import MISSING
"""
Fusion Layers

Fuses vision-language features with proprioception.

TODO Phase 2 (Week 2, Priority P0):
- [ ] Implement concat + MLP fusion
- [ ] Add cross-attention fusion (optional)
- [ ] Support different fusion strategies
"""

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.mlp import MLP


@configclass
class FusionLayerCfg(ModuleBaseCfg):
    """
    Fusion layer configuration.

    Fusion strategies:
    - "concat_mlp": Concatenate VL + proprio, then MLP
    - "cross_attention": Cross-attention between VL and proprio (future)
    """

    class_type: type["FusionLayer"] = MISSING

    fusion_type: str = "concat_mlp"
    output_dim: int = 512
    hidden_dims: list[int] = [512]
    activation: str = "relu"


class FusionLayer(ModuleBase):
    """
    Fuses VL features with proprioception.

    Input:
    - vl_features: (B, vl_dim) from VLM backbone
    - proprioception: (B, proprio_dim) from robot

    Output:
    - fused_features: (B, output_dim) ready for action head
    """

    def __init__(self, cfg: FusionLayerCfg, dim_params: dict):
        super().__init__(cfg)

        vl_dim = dim_params["vl_feature_dim"]
        proprio_dim = dim_params["proprio_dim"]

        if cfg.fusion_type == "concat_mlp":
            # TODO: Implement concat + MLP fusion
            # input_dim = vl_dim + proprio_dim
            # self.mlp = MLP(
            #     input_dim=input_dim,
            #     output_dim=cfg.output_dim,
            #     hidden_dims=cfg.hidden_dims,
            #     activation=cfg.activation,
            # )
            raise NotImplementedError("TODO: Implement concat_mlp fusion")

        elif cfg.fusion_type == "cross_attention":
            # TODO: Implement cross-attention fusion (future)
            raise NotImplementedError("TODO: Implement cross_attention fusion")

        else:
            raise ValueError(f"Unknown fusion_type: {cfg.fusion_type}")

        self.output_dim = cfg.output_dim

    def forward(
        self,
        vl_features: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse VL features with proprioception.

        Args:
            vl_features: (B, vl_dim)
            proprioception: (B, proprio_dim)

        Returns:
            fused: (B, output_dim)

        TODO:
        - Concatenate inputs
        - Pass through MLP
        - Return fused features
        """
        raise NotImplementedError("TODO: Implement fusion forward pass")

        # Example:
        # concat = torch.cat([vl_features, proprioception], dim=-1)
        # fused = self.mlp(concat)
        # return fused
