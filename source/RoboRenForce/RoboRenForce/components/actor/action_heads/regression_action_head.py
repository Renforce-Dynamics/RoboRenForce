"""
Regression Action Head (Baseline)

Simple MLP-based action head for direct action regression.

IO Contract:
    Input:
        features: (B, input_dim) fused features from fusion layer

    Output:
        actions:  (B, action_horizon, action_dim) predicted actions

    Train forward (train_forward):
        Input:  features (B, input_dim), target_actions (B, action_horizon, action_dim)
        Output: dict {"pred_actions": ..., "target_actions": ...}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


class RegressionActionHead(ModuleBase):
    """
    MLP-based action head for direct regression.

    Output shape is always (B, action_horizon, action_dim).
    """

    def __init__(self, cfg: RegressionActionHeadCfg, dim_params: dict):
        super().__init__()
        self.cfg = cfg

        input_dim = dim_params["input_dim"]
        action_dim = cfg.action_dim
        action_horizon = cfg.action_horizon
        output_dim = action_dim * action_horizon

        layers = []
        dims = [input_dim] + list(cfg.hidden_dims) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if cfg.activation == "relu":
                    layers.append(nn.ReLU())
                elif cfg.activation == "gelu":
                    layers.append(nn.GELU())
                elif cfg.activation == "silu":
                    layers.append(nn.SiLU())
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
        self.mlp = nn.Sequential(*layers)

        self.action_dim = action_dim
        self.action_horizon = action_horizon

    def forward(self, features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        B = features.shape[0]
        out = self.mlp(features)
        return out.view(B, self.action_horizon, self.action_dim)

    def train_forward(self, features: torch.Tensor, target_actions: torch.Tensor) -> dict:
        pred_actions = self.forward(features)
        return {"pred_actions": pred_actions, "target_actions": target_actions}


@configclass
class RegressionActionHeadCfg(ModuleBaseCfg):
    """Simple MLP action head configuration."""

    class_type: type[RegressionActionHead] = RegressionActionHead

    hidden_dims: list = [256, 256]
    activation: str = "relu"
    dropout: float = 0.0
    action_dim: int = 0
    action_horizon: int = 1
