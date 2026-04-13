from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from dataclasses import MISSING

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.mlp import MLP, MLPCfg


class VqVae(ModuleBase):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE).
    
    Uses a codebook to quantize latent representations.
    """

    def __init__(
        self,
        cfg: VqVaeCfg,
        input_size: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        """
        Args:
            cfg: Configuration for the VQ-VAE.
            input_size: Input dimension.
            codebook_size: Number of codebook vectors.
            codebook_dim: Dimension of each codebook vector.
        """
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = cfg.commitment_cost
        
        # Build encoder: input -> codebook_dim
        encoder_cfg = cfg.encoder_cfg
        encoder_cfg = encoder_cfg.replace(
            hidden_features=encoder_cfg.hidden_features,
            activations=encoder_cfg.activations,
        )
        self.encoder = encoder_cfg.construct_from_cfg(
            in_feature=input_size,
            out_feature=codebook_dim
        )
        
        # Codebook: embedding table
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
        # Build decoder: codebook_dim -> input_size
        decoder_cfg = cfg.decoder_cfg
        decoder_cfg = decoder_cfg.replace(
            hidden_features=decoder_cfg.hidden_features,
            activations=decoder_cfg.activations,
        )
        self.decoder = decoder_cfg.construct_from_cfg(
            in_feature=codebook_dim,
            out_feature=input_size
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VQ-VAE.
        
        Args:
            x: Input tensor of shape (..., input_size)
            
        Returns:
            Tuple of (reconstructed, quantized, loss, indices)
            - reconstructed: (..., input_size)
            - quantized: (..., codebook_dim)
            - loss: scalar tensor (only non-zero during training)
            - indices: (..., 1) codebook indices
        """
        # Encode: (..., input_size) -> (..., codebook_dim)
        z = self.encoder(x)
        
        # Compute distances to codebook vectors
        # z: (..., codebook_dim), codebook.weight: (codebook_size, codebook_dim)
        # distances: (..., codebook_size)
        distances = (
            torch.sum(z**2, dim=-1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.einsum('...d,nd->...n', z, self.codebook.weight)
        )
        
        # Find nearest codebook vector
        indices = torch.argmin(distances, dim=-1).unsqueeze(-1)  # (..., 1)
        
        # Create one-hot encodings
        encodings = torch.zeros(
            indices.shape[:-1] + (self.codebook_size,),
            device=z.device
        )
        encodings.scatter_(-1, indices, 1)
        
        # Quantize: (..., codebook_dim)
        quantized = torch.matmul(encodings, self.codebook.weight).view(z.shape)
        
        # Compute loss (only during training)
        if self.training:
            e_latent_loss = F.mse_loss(quantized.detach(), z)
            q_latent_loss = F.mse_loss(quantized, z.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        else:
            loss = torch.tensor(0.0, device=z.device)
        
        # Straight-through estimator: use quantized in forward, but gradients flow to z
        quantized = z + (quantized - z).detach()
        
        # Decode: (..., codebook_dim) -> (..., input_size)
        reconstructed = self.decoder(quantized)
        
        return reconstructed, quantized, loss, indices


@configclass
class VqVaeCfg(ModuleBaseCfg):
    """Configuration for VqVae."""
    class_type: type[nn.Module] = VqVae
    
    encoder_cfg: MLPCfg = MISSING
    """Configuration for the encoder MLP."""
    
    decoder_cfg: MLPCfg = MISSING
    """Configuration for the decoder MLP."""
    
    commitment_cost: float = 0.25
    """Commitment cost for VQ-VAE loss."""
