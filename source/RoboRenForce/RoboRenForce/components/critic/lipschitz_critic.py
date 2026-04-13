from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from dataclasses import MISSING

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.mlp import MLP, MLPCfg
from RoboRenForce.networks.fft_filter import FFTFilter1D, FFTFilter1DCfg, FFTFilter2D, FFTFilter2DCfg


class LipschitzCritic(ModuleBase):
    """
    Lipschitz-constrained critic with optional FFT filtering.
    
    This critic supports FFT-based observation filtering for value estimation.
    """

    def __init__(
        self,
        cfg: LipschitzCriticCfg,
        state_dim: int,
        out_feature: int = 1,
    ):
        """
        Args:
            cfg: Configuration for the critic.
            state_dim: Input state dimension.
            out_feature: Output dimension (usually 1 for scalar value).
        """
        super().__init__()
        self.cfg = cfg
        self.state_dim = state_dim
        self.out_feature = out_feature
        
        # FFT filter (optional)
        self.enable_fft = cfg.enable_fft
        self.fft_filter = None
        if self.enable_fft:
            if cfg.fft_2d:
                fft_cfg = cfg.fft_filter_2d_cfg if cfg.fft_filter_2d_cfg is not None else FFTFilter2DCfg()
                self.fft_filter = fft_cfg.construct_from_cfg(
                    seq_len=cfg.obs_seq_len,
                    feature_dim=state_dim
                )
            else:
                fft_cfg = cfg.fft_filter_1d_cfg if cfg.fft_filter_1d_cfg is not None else FFTFilter1DCfg()
                self.fft_filter = fft_cfg.construct_from_cfg(
                    seq_len=cfg.obs_seq_len,
                    feature_dim=state_dim
                )

        # Critic backbone
        self.backbone: MLP = cfg.backbone_cfg.class_type(
            in_feature=state_dim,
            out_feature=out_feature,
            cfg=cfg.backbone_cfg
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute value estimate.
        
        Args:
            state: Input state tensor of shape (B, state_dim) or (B, L, state_dim)
            
        Returns:
            Value estimate of shape (B, out_feature)
        """
        # Apply FFT filter if enabled
        if self.enable_fft and self.fft_filter is not None:
            # If sequence input, filter and take last timestep
            if len(state.shape) == 3:
                state_filtered = self.fft_filter(state)
                state = state_filtered[:, -1, :]  # Take most recent
            else:
                state_filtered = self.fft_filter(state.unsqueeze(1))
                state = state_filtered.squeeze(1)
        
        # Compute value
        value = self.backbone(state)
        return value

    def reset(self, *args, **kwargs):
        """Reset internal state (no-op for this critic)."""
        pass


@configclass
class LipschitzCriticCfg(ModuleBaseCfg):
    """Configuration for LipschitzCritic."""
    class_type: type[nn.Module] = LipschitzCritic
    
    backbone_cfg: MLPCfg = MISSING
    """Configuration for the critic backbone MLP."""
    
    enable_fft: bool = False
    """Whether to enable FFT filtering."""
    
    fft_2d: bool = True
    """Whether to use 2D FFT (spatio-temporal) or 1D FFT (temporal only)."""
    
    obs_seq_len: int = 1
    """Observation sequence length for FFT filtering."""
    
    fft_filter_1d_cfg: Optional[FFTFilter1DCfg] = None
    """Configuration for 1D FFT filter (used if fft_2d=False)."""
    
    fft_filter_2d_cfg: Optional[FFTFilter2DCfg] = None
    """Configuration for 2D FFT filter (used if fft_2d=True)."""
