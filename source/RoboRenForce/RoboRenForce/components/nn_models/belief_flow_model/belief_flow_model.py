from __future__ import annotations

import torch
import torch.nn as nn
from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.mlp import MLPCfg


class BeliefFlowModel(ModuleBase):
    """
    Belief flow model: f_theta(h, low_obs, action) -> dh

    Predicts the change in belief state given:
    - h: current belief state [belief_dim]
    - low_obs: low-frequency observation [policy_dim]
    - action: previous action [action_dim]

    Output: dh [belief_dim] - change in belief state
    """

    def __init__(
        self,
        cfg: "BeliefFlowModelCfg",
        belief_dim: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg
        self.belief_dim = belief_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Input: [h, low_obs, action]
        input_dim = belief_dim + obs_dim + action_dim

        # MLP backbone
        self.mlp = cfg.backbone_cfg.class_type(
            in_feature=input_dim,
            out_feature=belief_dim,
            cfg=cfg.backbone_cfg,
        )

        self.to(device)

    def forward(
        self,
        h: torch.Tensor,
        low_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            h: Belief state [..., belief_dim]
            low_obs: Low-frequency observation [..., obs_dim]
            action: Previous action [..., action_dim]

        Returns:
            dh: Change in belief state [..., belief_dim]
        """
        x = torch.cat([h, low_obs, action], dim=-1)
        return self.mlp(x)


@configclass
class BeliefFlowModelCfg(ModuleBaseCfg):
    """Configuration for BeliefFlowModel."""

    class_type: type[nn.Module] = BeliefFlowModel

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

        return BeliefFlowModel(
            cfg=self,
            belief_dim=dim_params.get("belief_dim", 64),
            obs_dim=dim_params["policy_dim"],
            action_dim=dim_params["action_dim"],
            device=device,
        )
