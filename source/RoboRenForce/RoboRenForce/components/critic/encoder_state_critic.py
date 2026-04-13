from __future__ import annotations

from dataclasses import MISSING
from typing import Dict

import torch
import torch.nn as nn

from RoboRenForce import configclass
from RoboRenForce.components.critic.v_network import VNetwork, VNetworkCfg
from RoboRenForce.components.encoder.vec_state_encoder import VecStateEncoder, VecStateEncoderCfg
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


class EncoderStateCritic(ModuleBase):
    """
    Critic that first encodes critic observations with a VecStateEncoder and then
    applies a value network (VNetwork) on the encoded critic state.
    """

    def __init__(
        self,
        cfg: "EncoderStateCriticCfg",
        critic_obs_dim: int,
        out_feature: int = 1,
    ):
        super().__init__()

        self.cfg = cfg

        # Encoder: critic_obs_dim -> encoded_dim
        self.encoder: VecStateEncoder = cfg.encoder_cfg.class_type(
            cfg=cfg.encoder_cfg,
            in_feature=critic_obs_dim,
            out_feature=cfg.encoded_dim,
        )

        # Underlying value network on encoded critic state
        self.vnet: VNetwork = cfg.critic_cfg.class_type(
            cfg=cfg.critic_cfg,
            state_dim=cfg.encoded_dim,
            out_feature=out_feature,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(state)
        return self.vnet(encoded)

    def reset(self, *args, **kwargs):
        self.vnet.reset(*args, **kwargs)


@configclass
class EncoderStateCriticCfg(ModuleBaseCfg):
    """Configuration for EncoderStateCritic."""

    class_type: type[nn.Module] = EncoderStateCritic

    # Dimension of encoded critic observation
    encoded_dim: int = 256

    # Sub-configs
    encoder_cfg: VecStateEncoderCfg = MISSING
    critic_cfg: VNetworkCfg = VNetworkCfg()

    def construct_from_cfg(self, *args, dim_params: Dict = None, **kwargs):
        if dim_params is None:
            return super().construct_from_cfg(*args, **kwargs)
        return EncoderStateCritic(
            cfg=self,
            critic_obs_dim=dim_params["critic_dim"],
            out_feature=1,
        )

