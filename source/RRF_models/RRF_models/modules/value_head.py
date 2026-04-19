"""
Value Head — MLP value estimator attached to policy backbone.

Used by PPO and other actor-critic RL algorithms.
Takes hidden state from backbone → scalar value estimate.

Reference: RLinf rlinf/models/embodiment/modules/value_head.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """MLP value head: backbone features → scalar value."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 128),
        activation: str = "gelu",
    ):
        super().__init__()
        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), act_fn()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, feature_dim] from backbone

        Returns:
            values: [B, 1]
        """
        return self.mlp(features)
