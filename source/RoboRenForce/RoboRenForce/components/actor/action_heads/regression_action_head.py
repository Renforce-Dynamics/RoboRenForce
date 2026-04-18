from dataclasses import MISSING
"""
Regression Action Head (Baseline)

Simple MLP-based action head for direct regression (baseline).

TODO Phase 2 (Week 2, Priority P1):
- [ ] Implement MLP action head
- [ ] Support different activations
- [ ] Add dropout (optional)
"""

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.mlp import MLP


@configclass
class RegressionActionHeadCfg(ModuleBaseCfg):
    """Simple MLP action head configuration."""

    class_type: type["RegressionActionHead"] = MISSING

    hidden_dims: list[int] = [256, 256]
    activation: str = "relu"
    dropout: float = 0.0
    action_dim: int = MISSING


class RegressionActionHead(ModuleBase):
    """
    MLP-based action head for direct regression.

    Simpler alternative to diffusion head, useful for:
    - Baseline comparisons
    - Fast prototyping
    - Tasks where diffusion is overkill

    TODO Phase 2:
    - [ ] Build MLP from input_dim to action_dim
    - [ ] Add dropout if specified
    - [ ] Implement forward pass
    """

    def __init__(self, cfg: RegressionActionHeadCfg, dim_params: dict):
        super().__init__(cfg)

        input_dim = dim_params["input_dim"]
        action_dim = cfg.action_dim

        # TODO: Build MLP
        # self.mlp = MLP(
        #     input_dim=input_dim,
        #     output_dim=action_dim,
        #     hidden_dims=cfg.hidden_dims,
        #     activation=cfg.activation,
        #     dropout=cfg.dropout,
        # )
        raise NotImplementedError("TODO: Build MLP action head")

    def forward(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Predict actions directly.

        Args:
            features: (B, input_dim) fused features
            deterministic: Ignored (no stochasticity in regression)

        Returns:
            actions: (B, action_dim)

        TODO:
        - Pass features through MLP
        - Return predicted actions
        """
        raise NotImplementedError("TODO: Implement regression forward")

        # Example:
        # return self.mlp(features)
