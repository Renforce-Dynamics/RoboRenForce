from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as D
from typing import Optional
from dataclasses import MISSING

from RoboRenForce import configclass
from RoboRenForce.networks.mlp import MLPCfg
from RoboRenForce.networks.fft_filter import FFTFilter1D, FFTFilter1DCfg, FFTFilter2D, FFTFilter2DCfg
from RoboRenForce.utils.template.module_base import ModuleBaseCfg
from RoboRenForce.components.actor.actor_base import ActorBase


class LipschitzActor(ActorBase):
    """
    Lipschitz-constrained actor with optional FFT filtering.
    
    This actor supports:
    - FFT-based observation filtering (1D or 2D)
    - State-independent action noise
    - Jacobian regularization (computed externally)
    """

    def __init__(
        self,
        cfg: LipschitzActorCfg,
        state_dim: int,
        action_dim: int,
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim)

        self.cfg = cfg
        
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

        # Actor backbone
        self.backbone = cfg.backbone_cfg.class_type(
            in_feature=state_dim,
            out_feature=action_dim,
            cfg=cfg.backbone_cfg,
        )

        # Action noise (state-independent)
        self.std = nn.Parameter(torch.ones(action_dim) * cfg.init_noise_std)
        
        # Disable args validation for speedup
        D.Normal.set_default_validate_args = False
        
        self.act_dist: Optional[D.Distribution] = None

    def forward(self, state: torch.Tensor) -> D.Distribution:
        """
        Forward pass: compute action distribution.
        
        Args:
            state: Input state tensor of shape (B, state_dim) or (B, L, state_dim)
            
        Returns:
            Action distribution
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
        
        # Compute mean
        mean = self.backbone(state)
        
        # Create distribution with state-independent std
        std = self.std.expand_as(mean)
        dist = D.Normal(mean, std)
        
        self.act_dist = dist
        return dist

    @torch.no_grad()
    def act_inference(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for inference/play.
        
        Args:
            state: Input state tensor
            
        Returns:
            Deterministic action (mean)
        """
        dist = self(state)
        return dist.mean

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample action from distribution."""
        dist = self(state)
        if deterministic:
            return dist.mean
        return dist.sample()

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """Sample action from the policy (PPO-compatible API)."""
        dist = self(state)
        return dist.sample()

    def get_actions_log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Compute log-probability of given actions under the cached distribution."""
        if self.act_dist is None:
            raise ValueError("Must call forward() or act() before get_actions_log_prob()")
        return self.act_dist.log_prob(action).sum(dim=-1)

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action given state."""
        dist = self(state)
        return dist.log_prob(action).sum(dim=-1)

    def entropy(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute entropy of action distribution."""
        dist = self(state) if state is not None else self.act_dist
        if dist is None:
            raise ValueError("Must call forward() or sample() before entropy()")
        return dist.entropy().sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        """Get mean of current action distribution."""
        if self.act_dist is None:
            raise ValueError("Must call forward() before accessing action_mean")
        return self.act_dist.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Get std of current action distribution."""
        if self.act_dist is None:
            raise ValueError("Must call forward() before accessing action_std")
        return self.act_dist.stddev

    def reset(self, *args, **kwargs):
        """Reset internal state (no-op for this actor)."""
        pass


@configclass
class LipschitzActorCfg(ModuleBaseCfg):
    """Configuration for LipschitzActor."""
    class_type: type[nn.Module] = LipschitzActor
    
    backbone_cfg: MLPCfg = MISSING
    """Configuration for the actor backbone MLP."""
    
    init_noise_std: float = 1.0
    """Initial standard deviation for action noise."""
    
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

    def construct_from_cfg(self, *args, dim_params: dict = None, **kwargs):
        """Construct LipschitzActor from configuration with dimension parameters."""
        if dim_params is None:
            return super().construct_from_cfg(*args, **kwargs)
        
        return LipschitzActor(
            self,
            dim_params["policy_dim"],
            dim_params["action_dim"],
        )
