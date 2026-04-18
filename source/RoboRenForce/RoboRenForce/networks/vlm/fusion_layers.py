"""
Fusion Layers

Fuses vision-language features with proprioception.

IO Contract:
    Input:
        vl_features:    (B, vl_dim)     from VLM backbone
        proprioception: (B, proprio_dim) from robot state

    Output:
        fused_features: (B, output_dim) ready for action head
"""

from __future__ import annotations

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


class FusionLayer(ModuleBase):
    """
    Fuses VL features with proprioception via concat + MLP.

    Input:
    - vl_features: (B, vl_dim) from VLM backbone
    - proprioception: (B, proprio_dim) from robot

    Output:
    - fused_features: (B, output_dim) ready for action head
    """

    def __init__(self, cfg: FusionLayerCfg, dim_params: dict):
        super().__init__()
        self.cfg = cfg

        vl_dim = dim_params["vl_feature_dim"]
        proprio_dim = dim_params.get("proprio_dim", 0)

        self.output_dim = cfg.output_dim

        if cfg.fusion_type == "concat_mlp":
            input_dim = vl_dim + proprio_dim
            layers = []
            dims = [input_dim] + list(cfg.hidden_dims) + [cfg.output_dim]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:  # no activation on last layer
                    if cfg.activation == "relu":
                        layers.append(nn.ReLU())
                    elif cfg.activation == "gelu":
                        layers.append(nn.GELU())
                    elif cfg.activation == "silu":
                        layers.append(nn.SiLU())
                    if cfg.dropout > 0:
                        layers.append(nn.Dropout(cfg.dropout))
            self.mlp = nn.Sequential(*layers)

        elif cfg.fusion_type == "add":
            self.vl_proj = nn.Linear(vl_dim, cfg.output_dim)
            self.proprio_proj = nn.Linear(proprio_dim, cfg.output_dim)
            self.layer_norm = nn.LayerNorm(cfg.output_dim)

        else:
            raise ValueError(f"Unknown fusion_type: {cfg.fusion_type}")

    def forward(
        self,
        vl_features: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.fusion_type == "concat_mlp":
            concat = torch.cat([vl_features, proprioception], dim=-1)
            return self.mlp(concat)
        elif self.cfg.fusion_type == "add":
            vl_proj = self.vl_proj(vl_features)
            proprio_proj = self.proprio_proj(proprioception)
            return self.layer_norm(vl_proj + proprio_proj)


@configclass
class FusionLayerCfg(ModuleBaseCfg):
    """Fusion layer configuration."""

    class_type: type[FusionLayer] = FusionLayer

    fusion_type: str = "concat_mlp"
    output_dim: int = 512
    hidden_dims: list = [512]
    activation: str = "relu"
    dropout: float = 0.0
