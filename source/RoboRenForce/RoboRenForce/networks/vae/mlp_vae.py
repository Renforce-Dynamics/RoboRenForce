from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Optional, Tuple
from dataclasses import MISSING

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.mlp import MLP, MLPCfg


class MlpVae(ModuleBase):
    """
    A simple MLP-based Variational Autoencoder (VAE).
    
    The encoder outputs mean and log_std for a latent distribution,
    and the decoder reconstructs from sampled latents.
    """

    def __init__(
        self,
        cfg: MlpVaeCfg,
        input_size: int,
        latent_size: int,
        decoder_aux_input_size: int = 0,
    ):
        """
        Args:
            cfg: Configuration for the VAE.
            input_size: Input dimension.
            latent_size: Latent dimension.
            decoder_aux_input_size: Additional input size for decoder (e.g., action).
        """
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size
        self.latent_size = latent_size
        self.decoder_aux_input_size = decoder_aux_input_size
        
        # Build encoder: input -> latent_size * 2 (mean + log_std)
        encoder_cfg = cfg.encoder_cfg
        encoder_cfg = encoder_cfg.replace(
            hidden_features=encoder_cfg.hidden_features,
            activations=encoder_cfg.activations,
        )
        self.encoder = encoder_cfg.construct_from_cfg(
            in_feature=input_size,
            out_feature=latent_size * 2
        )
        
        # Build decoder: (latent_size + decoder_aux_input_size) -> output_size
        decoder_cfg = cfg.decoder_cfg
        decoder_cfg = decoder_cfg.replace(
            hidden_features=decoder_cfg.hidden_features,
            activations=decoder_cfg.activations,
        )
        decoder_input_size = latent_size + decoder_aux_input_size
        self.decoder = decoder_cfg.construct_from_cfg(
            in_feature=decoder_input_size,
            out_feature=input_size
        )

    def forward(
        self, 
        x: torch.Tensor, 
        decoder_aux_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dist.Distribution]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor of shape (..., input_size)
            decoder_aux_input: Optional auxiliary input for decoder of shape (..., decoder_aux_input_size)
            
        Returns:
            Tuple of (reconstructed, latent_distribution)
        """
        # Encode: (..., input_size) -> (..., latent_size * 2)
        encoder_out = self.encoder(x)
        z_mean, z_log_std = encoder_out.chunk(2, dim=-1)
        
        # Sample from latent distribution
        z = z_mean + z_log_std.exp() * torch.randn_like(z_mean)
        
        # Add auxiliary input if provided
        if decoder_aux_input is not None:
            z = torch.cat([z, decoder_aux_input], dim=-1)
        
        # Decode: (..., latent_size + aux) -> (..., input_size)
        reconstructed = self.decoder(z)
        
        # Create latent distribution
        latent_dist = dist.Normal(z_mean, z_log_std.exp())
        
        return reconstructed, latent_dist


@configclass
class MlpVaeCfg(ModuleBaseCfg):
    """Configuration for MlpVae."""
    class_type: type[nn.Module] = MlpVae
    
    encoder_cfg: MLPCfg = MISSING
    """Configuration for the encoder MLP."""
    
    decoder_cfg: MLPCfg = MISSING
    """Configuration for the decoder MLP."""
