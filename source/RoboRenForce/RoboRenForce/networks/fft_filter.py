from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal, Optional
from dataclasses import MISSING

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


class FFTFilter1D(ModuleBase):
    """
    1D FFT filter for temporal sequences.
    
    Applies a learnable filter in the frequency domain to input sequences.
    """

    def __init__(
        self,
        cfg: FFTFilter1DCfg,
        seq_len: int,
        feature_dim: int,
    ):
        """
        Args:
            cfg: Configuration for the FFT filter.
            seq_len: Sequence length.
            feature_dim: Feature dimension.
        """
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.kernel_scale = cfg.kernel_scale
        
        # Initialize filter kernel in frequency domain
        # rfft output size: seq_len//2 + 1
        kernel_size = (seq_len // 2 + 1, feature_dim, 2)  # (freq, features, real/imag)
        
        # Initialize: real part = 1, imag part = small random
        kernel_real = torch.ones(seq_len // 2 + 1, feature_dim, 1, dtype=torch.float32)
        kernel_imag = torch.randn(seq_len // 2 + 1, feature_dim, 1, dtype=torch.float32) * self.kernel_scale
        
        self.filter_kernel = nn.Parameter(
            torch.cat([kernel_real, kernel_imag], dim=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FFT filter to input sequence.
        
        Args:
            x: Input tensor of shape (B, seq_len, feature_dim) or (B, feature_dim)
            
        Returns:
            Filtered tensor of same shape as input
        """
        # Handle 2D input (single timestep) - pad to seq_len
        original_shape = x.shape
        if len(x.shape) == 2:
            # Single timestep: (B, feature_dim) -> (B, 1, feature_dim)
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            # Sequence: (B, L, feature_dim)
            squeeze_output = False
        
        # Pad or truncate to seq_len if needed
        current_seq_len = x.shape[1]
        if current_seq_len < self.seq_len:
            # Pad with zeros
            padding = torch.zeros(x.shape[0], self.seq_len - current_seq_len, x.shape[2], 
                                device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        elif current_seq_len > self.seq_len:
            # Truncate
            x = x[:, :self.seq_len, :]
        
        # FFT: (B, seq_len, feature_dim) -> (B, seq_len//2+1, feature_dim) complex
        x_f = torch.fft.rfft(x, n=self.seq_len, dim=1, norm='ortho')
        
        # Apply filter: multiply with complex kernel
        kernel = torch.view_as_complex(self.filter_kernel)  # (seq_len//2+1, feature_dim)
        x_f = x_f * kernel.unsqueeze(0)  # Broadcast: (B, seq_len//2+1, feature_dim)
        
        # IFFT: back to time domain
        x_filtered = torch.fft.irfft(x_f, n=self.seq_len, dim=1, norm='ortho')
        
        # Restore original sequence length
        if current_seq_len < self.seq_len:
            x_filtered = x_filtered[:, :current_seq_len, :]
        elif current_seq_len > self.seq_len:
            # This shouldn't happen after truncation, but handle it
            pass
        
        if squeeze_output:
            x_filtered = x_filtered.squeeze(1)
        
        return x_filtered


@configclass
class FFTFilter1DCfg(ModuleBaseCfg):
    """Configuration for FFTFilter1D."""
    class_type: type[nn.Module] = FFTFilter1D
    
    kernel_scale: float = 0.02
    """Scale for initializing imaginary part of filter kernel."""


class FFTFilter2D(ModuleBase):
    """
    2D FFT filter for spatio-temporal sequences.
    
    Applies a learnable 2D filter in the frequency domain.
    """

    def __init__(
        self,
        cfg: FFTFilter2DCfg,
        seq_len: int,
        feature_dim: int,
    ):
        """
        Args:
            cfg: Configuration for the FFT filter.
            seq_len: Sequence length (temporal dimension).
            feature_dim: Feature dimension (spatial dimension).
        """
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.kernel_scale = cfg.kernel_scale
        
        # Initialize filter kernel in frequency domain
        # rfft2 output size: (seq_len, feature_dim//2 + 1)
        kernel_size = (seq_len, feature_dim // 2 + 1, 2)  # (time_freq, space_freq, real/imag)
        
        # Initialize: real part = 1, imag part = small random
        kernel_real = torch.ones(seq_len, feature_dim // 2 + 1, 1, dtype=torch.float32)
        kernel_imag = torch.randn(seq_len, feature_dim // 2 + 1, 1, dtype=torch.float32) * self.kernel_scale
        
        self.filter_kernel = nn.Parameter(
            torch.cat([kernel_real, kernel_imag], dim=2)
        )
        
        # Optional normalization layer
        self.norm_layer = None
        if cfg.norm_layer_type == "batch_norm":
            self.norm_layer = nn.BatchNorm2d(feature_dim)
        elif cfg.norm_layer_type == "layer_norm":
            self.norm_layer = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D FFT filter to input sequence.
        
        Args:
            x: Input tensor of shape (B, seq_len, feature_dim) or (B, feature_dim)
            
        Returns:
            Filtered tensor of same shape as input
        """
        # Handle 2D input (single timestep) - pad to seq_len
        original_shape = x.shape
        if len(x.shape) == 2:
            # Single timestep: (B, feature_dim) -> (B, 1, feature_dim)
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            # Sequence: (B, L, feature_dim)
            squeeze_output = False
        
        # Pad or truncate to seq_len if needed
        current_seq_len = x.shape[1]
        if current_seq_len < self.seq_len:
            # Pad with zeros
            padding = torch.zeros(x.shape[0], self.seq_len - current_seq_len, x.shape[2], 
                                device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        elif current_seq_len > self.seq_len:
            # Truncate
            x = x[:, :self.seq_len, :]
        
        # Apply normalization if specified
        if self.norm_layer is not None:
            if isinstance(self.norm_layer, nn.BatchNorm2d):
                # Reshape for BatchNorm2d: (B, seq_len, feature_dim) -> (B*seq_len, feature_dim) -> (B, feature_dim, 1, seq_len)
                x_reshaped = x.reshape(-1, self.feature_dim).unsqueeze(1).unsqueeze(2)
                x_norm = self.norm_layer(x_reshaped).squeeze(2).squeeze(1).reshape(x.shape)
            else:  # LayerNorm
                x_norm = self.norm_layer(x)
        else:
            x_norm = x
        
        # 2D FFT: (B, seq_len, feature_dim) -> (B, seq_len, feature_dim//2+1) complex
        x_f = torch.fft.rfft2(x_norm, s=(self.seq_len, self.feature_dim), dim=(1, 2), norm='ortho')
        
        # Apply filter: multiply with complex kernel
        kernel = torch.view_as_complex(self.filter_kernel)  # (seq_len, feature_dim//2+1)
        x_f = x_f * kernel.unsqueeze(0)  # Broadcast: (B, seq_len, feature_dim//2+1)
        
        # IFFT: back to time domain
        x_filtered = torch.fft.irfft2(x_f, s=(self.seq_len, self.feature_dim), dim=(1, 2), norm='ortho')
        
        # Restore original sequence length
        if current_seq_len < self.seq_len:
            x_filtered = x_filtered[:, :current_seq_len, :]
        elif current_seq_len > self.seq_len:
            # This shouldn't happen after truncation, but handle it
            pass
        
        if squeeze_output:
            x_filtered = x_filtered.squeeze(1)
        
        return x_filtered


@configclass
class FFTFilter2DCfg(ModuleBaseCfg):
    """Configuration for FFTFilter2D."""
    class_type: type[nn.Module] = FFTFilter2D
    
    kernel_scale: float = 0.02
    """Scale for initializing imaginary part of filter kernel."""
    
    norm_layer_type: Literal["none", "batch_norm", "layer_norm"] = "none"
    """Type of normalization layer to apply before FFT."""
