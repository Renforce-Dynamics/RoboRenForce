from __future__ import annotations

import torch
import torch.nn as nn
from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.mlp import MLPCfg


class BeliefUpdater(ModuleBase):
    """
    Belief observation update model: g_phi(h_pred, high_obs) -> Δh

    Predicts the correction to belief state given:
    - h_pred: predicted belief state [belief_dim]
    - high_obs: high-frequency observation [high_obs_dim]

    Output: Δh [belief_dim] - correction to belief state
    """

    def __init__(
        self,
        cfg: "BeliefUpdaterCfg",
        belief_dim: int,
        high_obs_dim: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg
        self.belief_dim = belief_dim
        self.high_obs_dim = high_obs_dim

        # Input: [h_pred, high_obs]
        input_dim = belief_dim + high_obs_dim

        # MLP backbone
        self.mlp = cfg.backbone_cfg.class_type(
            in_feature=input_dim,
            out_feature=belief_dim,
            cfg=cfg.backbone_cfg,
        )

        self.to(device)

    def forward(
        self,
        h_pred: torch.Tensor,
        high_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            h_pred: Predicted belief state [..., belief_dim]
            high_obs: High-frequency observation [..., high_obs_dim]

        Returns:
            Δh: Correction to belief state [..., belief_dim]
        """
        x = torch.cat([h_pred, high_obs], dim=-1)
        return self.mlp(x)


@configclass
class BeliefUpdaterCfg(ModuleBaseCfg):
    """Configuration for BeliefObsUpdate."""

    class_type: type[nn.Module] = BeliefUpdater

    # MLP backbone configuration
    backbone_cfg: MLPCfg = MLPCfg(
        hidden_features=[256, 256],
        activations=[
            [("ReLU", {})],
            [("ReLU", {})],
            [],  # No activation on last layer
        ],
    )

    def construct_from_cfg(
        self,
        *args,
        dim_params: dict = None,
        device: str = "cpu",
        **kwargs,
    ):
        if dim_params is None:
            return super().construct_from_cfg(*args, **kwargs)
    
        # For flow model training we use slow observations as the updater input.
        high_obs_dim = dim_params.get("slow_dim", dim_params.get("critic_dim", dim_params["policy_dim"]))
    
        return BeliefUpdater(
            cfg=self,
            belief_dim=dim_params.get("belief_dim", 64),
            high_obs_dim=high_obs_dim,
            device=device,
        )

