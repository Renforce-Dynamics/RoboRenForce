from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import MISSING

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.mlp import MLP, MLPCfg
from RoboRenForce.networks.activations import get_activation, ACTIVATION_TYPES


class MoeLayer(ModuleBase):
    """
    Mixture of Experts (MoE) layer.
    
    This layer consists of multiple expert networks and a gating network.
    The gating network determines how to combine the expert outputs.
    """

    def __init__(
        self,
        cfg: MoeLayerCfg,
        input_dim: int,
        output_dim: Optional[int] = None,
    ):
        """
        Args:
            cfg: Configuration for the MoE layer.
            input_dim: Input dimension.
            output_dim: Output dimension. If None, uses input_dim.
        """
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_experts = cfg.num_experts
        
        # Build gate network
        gate_cfg = cfg.gate_cfg
        if gate_cfg is None:
            # Default gate: simple MLP
            from RoboRenForce.networks.mlp import MLPCfg
            gate_cfg = MLPCfg(
                hidden_features=cfg.gate_hidden_dims,
                activations=[[('ReLU', {})] for _ in range(len(cfg.gate_hidden_dims))] + [[]],
            )
        
        # Gate outputs num_experts logits
        gate_out_dim = self.num_experts
        if cfg.gate_hidden_dims:
            gate_in_dim = input_dim
            gate_hidden = cfg.gate_hidden_dims
        else:
            # Direct linear layer
            gate_in_dim = input_dim
            gate_hidden = []
        
        # Build gate
        gate_final_cfg = gate_cfg.replace(
            hidden_features=gate_hidden,
            activations=gate_cfg.activations if gate_cfg.activations else [[('ReLU', {})] for _ in range(len(gate_hidden))] + [[]],
        )
        self.gate = gate_final_cfg.construct_from_cfg(
            in_feature=gate_in_dim,
            out_feature=gate_out_dim
        )
        
        # Build expert networks
        expert_cfg = cfg.expert_cfg
        if expert_cfg is None:
            # Default expert: simple MLP
            from RoboRenForce.networks.mlp import MLPCfg
            expert_cfg = MLPCfg(
                hidden_features=cfg.expert_hidden_dims,
                activations=[[('ReLU', {})] for _ in range(len(cfg.expert_hidden_dims))] + [[]],
            )
        
        self.experts = nn.ModuleList([
            expert_cfg.construct_from_cfg(
                in_feature=input_dim,
                out_feature=self.output_dim
            )
            for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        # Get gate scores: (..., num_experts)
        gate_scores = F.softmax(self.gate(x), dim=-1)
        
        # Get expert outputs: list of (..., output_dim)
        expert_outputs = [expert(x) for expert in self.experts]
        # Stack: (..., num_experts, output_dim)
        expert_outputs = torch.stack(expert_outputs, dim=-2)
        
        # Mix expert outputs: (..., output_dim)
        # gate_scores: (..., num_experts), expert_outputs: (..., num_experts, output_dim)
        output = torch.einsum('...e,...ed->...d', gate_scores, expert_outputs)
        
        return output


@configclass
class MoeLayerCfg(ModuleBaseCfg):
    """Configuration for MoeLayer."""
    class_type: type[nn.Module] = MoeLayer
    
    num_experts: int = MISSING
    """Number of expert networks."""
    
    expert_hidden_dims: List[int] = []
    """Hidden dimensions for each expert network. If empty, experts are linear."""
    
    expert_cfg: Optional[MLPCfg] = None
    """Configuration for expert networks. If None, uses expert_hidden_dims."""
    
    gate_hidden_dims: List[int] = []
    """Hidden dimensions for the gating network. If empty, gate is linear."""
    
    gate_cfg: Optional[MLPCfg] = None
    """Configuration for gating network. If None, uses gate_hidden_dims."""
