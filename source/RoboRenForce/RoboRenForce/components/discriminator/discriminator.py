import torch
import torch.nn as nn
from torch import autograd

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.utils.normalizer import RunningMeanStd
from RoboRenForce.networks.mlp import MLP, MLPCfg
from dataclasses import MISSING
from typing import Tuple


class Discriminator(ModuleBase):
    """Generic vector discriminator for adversarial imitation.

    This module reuses the shared MLP backbone infrastructure. It operates on
    pre-concatenated feature vectors (e.g., [s_t, s_{t+1}] or [s_t, a_t, s_{t+1}])
    and provides:
        - logits computation via an MLP backbone
        - optional gradient penalty
        - AMP-style reward shaping helper.
    """

    def __init__(self, cfg: "DiscriminatorCfg", input_dim: int, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.device = torch.device(device)

        # Split the backbone into ``trunk`` (input_dim → last_hidden) and
        # ``amp_linear`` (last_hidden → 1). AMPPPO applies different
        # weight-decay to each, so they need to be distinct submodules.
        hidden = list(cfg.backbone_cfg.hidden_features)
        if not hidden:
            raise ValueError(
                "DiscriminatorCfg.backbone_cfg.hidden_features must be "
                "non-empty (need at least one trunk hidden layer)."
            )
        last_hidden = hidden[-1]
        trunk_cfg = MLPCfg(
            hidden_features=hidden[:-1],
            activations=cfg.backbone_cfg.activations[:-1] or [[]],
        )
        self.trunk: MLP = MLP(
            cfg=trunk_cfg,
            in_feature=input_dim,
            out_feature=last_hidden,
        )
        self.amp_linear: nn.Linear = nn.Linear(last_hidden, 1)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns discriminator logits."""
        return self.amp_linear(self.trunk(x))

    def compute_grad_pen(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        lambda_: float = 10.0,
    ):
        """Gradient penalty on expert data, matching beyondAMP AMPDiscriminator.

        The gradient norm is encouraged towards 0 for stability, as in:
            grad_pen = λ * ||∇_x D(x)||_2^2
        where x = concat(expert_state, expert_next_state).
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad_(True)

        disc = self.forward(expert_data)
        ones = torch.ones_like(disc, device=disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0.0).pow(2).mean()
        return grad_pen


@configclass
class DiscriminatorCfg(ModuleBaseCfg):
    """Configuration for generic adversarial discriminator."""

    class_type: type[nn.Module] = Discriminator

    backbone_cfg: MLPCfg = MISSING

    def construct_from_cfg(self, input_dim: int, device: str = "cpu", *args, **kwargs):
        """Construct discriminator from configuration and input dimension."""
        return self.class_type(self, input_dim=input_dim, device=device)
