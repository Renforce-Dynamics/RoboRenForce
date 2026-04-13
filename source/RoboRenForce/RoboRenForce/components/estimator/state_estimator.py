from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from dataclasses import MISSING

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.mlp import MLP, MLPCfg
from RoboRenForce.networks.simple_rnn import SimpleRNN, SimpleRNNCfg


class StateEstimator(ModuleBase):
    """
    State estimator component that estimates missing or noisy state components.
    
    This component can be used to:
    - Estimate privileged information from observations
    - Denoise or complete partial observations
    - Predict state components that are not directly observable
    """

    def __init__(
        self,
        cfg: StateEstimatorCfg,
        input_dim: int,
        output_dim: int,
    ):
        """
        Args:
            cfg: Configuration for the state estimator.
            input_dim: Input observation dimension.
            output_dim: Output estimated state dimension.
        """
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_recurrent = cfg.use_rnn
        
        if self.is_recurrent:
            # Recurrent state estimator
            rnn_cfg = cfg.rnn_cfg if cfg.rnn_cfg is not None else SimpleRNNCfg()
            self.rnn = rnn_cfg.construct_from_cfg(input_dim=input_dim)
            
            # MLP head on top of RNN output
            estimator_cfg = cfg.estimator_cfg
            estimator_cfg = estimator_cfg.replace(
                hidden_features=estimator_cfg.hidden_features,
                activations=estimator_cfg.activations,
            )
            rnn_output_dim = rnn_cfg.hidden_size * (2 if rnn_cfg.bidirectional else 1)
            self.estimator = estimator_cfg.construct_from_cfg(
                in_feature=rnn_output_dim,
                out_feature=output_dim
            )
            self._hidden_state = None
        else:
            # Feedforward state estimator
            estimator_cfg = cfg.estimator_cfg
            estimator_cfg = estimator_cfg.replace(
                hidden_features=estimator_cfg.hidden_features,
                activations=estimator_cfg.activations,
            )
            self.estimator = estimator_cfg.construct_from_cfg(
                in_feature=input_dim,
                out_feature=output_dim
            )
            self.rnn = None
            self._hidden_state = None

    def forward(
        self, 
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through state estimator.
        
        Args:
            observations: Input observations of shape (B, input_dim) or (B, L, input_dim)
            
        Returns:
            estimated_state: (B, output_dim) or (B, L, output_dim)
        """
        if self.is_recurrent:
            # RNN forward: SimpleRNN returns (B, L, hidden_dim)
            rnn_out = self.rnn(observations)
            # Apply estimator head
            # If sequence, take last timestep or process all
            if len(rnn_out.shape) == 3:
                # Take last timestep for sequence input
                rnn_out = rnn_out[:, -1, :]  # (B, hidden_dim)
            estimated_state = self.estimator(rnn_out)
            return estimated_state
        else:
            # Feedforward
            estimated_state = self.estimator(observations)
            return estimated_state

    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """
        Reset hidden state for recurrent estimator.
        
        Args:
            batch_size: Batch size
            device: Device to create hidden state on
        """
        if self.is_recurrent and self.rnn is not None:
            if device is None:
                device = next(self.parameters()).device
            self.rnn.reset_hidden(batch_size=batch_size, device=device)


@configclass
class StateEstimatorCfg(ModuleBaseCfg):
    """Configuration for StateEstimator."""
    class_type: type[nn.Module] = StateEstimator
    
    estimator_cfg: MLPCfg = MISSING
    """Configuration for the estimator MLP network."""
    
    use_rnn: bool = False
    """Whether to use RNN for temporal modeling."""
    
    rnn_cfg: Optional[SimpleRNNCfg] = None
    """Configuration for RNN (only used if use_rnn=True)."""


class StateEstimatorWrapper(ModuleBase):
    """
    Wrapper that integrates state estimator with an actor.
    
    This wrapper estimates missing state components and optionally replaces
    them in the observations before passing to the actor.
    """

    def __init__(
        self,
        cfg: StateEstimatorWrapperCfg,
        actor: ModuleBase,
        obs_input_dim: int,
        obs_output_dim: int,
        estimated_dim: int,
    ):
        """
        Args:
            cfg: Configuration for the wrapper.
            actor: The actor network to wrap.
            obs_input_dim: Input observation dimension.
            obs_output_dim: Output observation dimension (after estimation).
            estimated_dim: Dimension of the estimated state component.
        """
        super().__init__()
        self.cfg = cfg
        self.actor = actor
        self.obs_input_dim = obs_input_dim
        self.obs_output_dim = obs_output_dim
        self.estimated_dim = estimated_dim
        self.replace_prob = cfg.replace_prob
        
        # Build state estimator
        estimator_input_dim = obs_input_dim - estimated_dim  # Assuming estimated part is removed from input
        self.estimator = cfg.estimator_cfg.construct_from_cfg(
            input_dim=estimator_input_dim,
            output_dim=estimated_dim
        )

    def forward(
        self, 
        observations: torch.Tensor,
        replace_estimated: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Forward pass: estimate state and optionally replace in observations.
        
        Args:
            observations: Input observations
            replace_estimated: Whether to replace estimated state. If None, uses cfg.replace_prob.
            
        Returns:
            Action from actor
        """
        # Estimate missing state
        # Extract input part (assuming estimated part is at the end and removed from input)
        input_part = observations[..., :self.obs_input_dim - self.estimated_dim]
        estimated_state = self.estimator(input_part)
        
        # Optionally replace in observations
        if replace_estimated is None:
            replace_estimated = torch.rand(1).item() < self.replace_prob
        
        if replace_estimated:
            # Replace estimated part in observations
            # This is a simplified version - actual implementation depends on observation structure
            obs_with_estimated = torch.cat([
                observations[..., :self.obs_input_dim - self.estimated_dim],
                estimated_state
            ], dim=-1)
        else:
            obs_with_estimated = observations
        
        # Pass to actor
        return self.actor(obs_with_estimated)

    @torch.no_grad()
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Inference mode: always use estimated state."""
        return self.forward(observations, replace_estimated=True)


@configclass
class StateEstimatorWrapperCfg(ModuleBaseCfg):
    """Configuration for StateEstimatorWrapper."""
    class_type: type[nn.Module] = StateEstimatorWrapper
    
    estimator_cfg: StateEstimatorCfg = MISSING
    """Configuration for the state estimator."""
    
    replace_prob: float = 0.0
    """Probability of replacing observations with estimated state during training."""
