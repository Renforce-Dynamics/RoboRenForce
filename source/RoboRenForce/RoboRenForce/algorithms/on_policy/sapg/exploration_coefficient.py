from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn
import math

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from dataclasses import MISSING


def create_sinusoidal_encoding(
    values: torch.Tensor, embd_size: int, n: int = 100
) -> torch.Tensor:
    """
    Create sinusoidal encoding for exploration coefficient embeddings.
    
    Args:
        values: Input values to encode [num_envs]
        embd_size: Embedding dimension
        n: Frequency parameter for sinusoidal encoding
        
    Returns:
        Encoded embeddings [num_envs, embd_size]
    """
    num_envs = values.shape[0]
    device = values.device
    
    # Create frequency matrix
    frequencies = torch.arange(1, embd_size + 1, device=device).float()
    frequencies = frequencies.unsqueeze(0)  # [1, embd_size]
    
    # Expand values
    values_expanded = values.unsqueeze(-1)  # [num_envs, 1]
    
    # Sinusoidal encoding
    angles = 2 * math.pi * frequencies * values_expanded / n
    encoding = torch.zeros(num_envs, embd_size, device=device)
    encoding[:, 0::2] = torch.sin(angles[:, 0::2])
    encoding[:, 1::2] = torch.cos(angles[:, 1::2])
    
    return encoding

class ExplorationCoefficient(ModuleBase):
    """
    Exploration coefficient manager for SAPG.
    
    Manages exploration coefficient embeddings and reward coefficients for different blocks.
    Each block uses a different exploration strategy to enable diverse exploration.
    """
    
    cfg: "ExplorationCoefficientCfg"
    def __init__(
        self,
        cfg: "ExplorationCoefficientCfg",
        num_envs: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        
        # Compute block size
        self.block_size = num_envs // cfg.num_blocks
        assert (
            num_envs % cfg.num_blocks == 0
        ), f"num_envs ({num_envs}) must be divisible by num_blocks ({cfg.num_blocks})"
        
        # Generate environment ID to block mapping
        # Each block contains block_size consecutive environments
        self.env_ids = torch.arange(cfg.num_blocks, device=device).repeat_interleave(
            self.block_size
        )
        
        # Initialize exploration coefficient embeddings
        self._init_embeddings()
        
        # Initialize reward coefficients
        self._init_reward_coefficients()
    
    def _init_embeddings(self):
        """Initialize exploration coefficient embeddings."""
        # Generate base embedding values (linearly spaced)
        embedding_genvec = torch.linspace(
            self.cfg.embd_init_range[0],
            self.cfg.embd_init_range[1],
            self.cfg.num_blocks,
            device=self.device,
        )[self.env_ids]  # [num_envs]
        
        if "learn_param" in self.cfg.expl_type:
            # Scalar embedding: [num_envs, 1]
            self.coef_embd = embedding_genvec.reshape(-1, 1)
        elif "disjoint" in self.cfg.expl_type:
            # Sinusoidal encoding embedding: [num_envs, embd_size]
            self.coef_embd = create_sinusoidal_encoding(
                embedding_genvec, self.cfg.embd_size
            )
        else:
            raise ValueError(f"Unknown expl_type: {self.cfg.expl_type}")
    
    def _init_reward_coefficients(self):
        """Initialize reward coefficients for intrinsic rewards."""
        if self.cfg.reward_coef_type == "entropy":
            # Linearly spaced reward coefficients
            self.reward_coef = (
                torch.linspace(
                    self.cfg.reward_coef_range[0],
                    self.cfg.reward_coef_range[1],
                    self.cfg.num_blocks,
                    device=self.device,
                )[self.env_ids]
                * self.cfg.reward_coef_scale
            )
        elif self.cfg.reward_coef_type == "none":
            # No intrinsic reward
            self.reward_coef = torch.zeros(self.num_envs, device=self.device)
        else:
            raise ValueError(
                f"Unknown reward_coef_type: {self.cfg.reward_coef_type}"
            )
    
    def get_embeddings_for_block(self, block_idx: int) -> torch.Tensor:
        """
        Get embeddings for a specific block by rolling the embedding tensor.
        
        Args:
            block_idx: Block index (0 to num_blocks - 1)
            
        Returns:
            Embeddings for the specified block [num_envs, embd_dim]
        """
        return torch.roll(self.coef_embd, self.block_size * block_idx, dims=0)
    
    def get_reward_coef_for_block(self, block_idx: int) -> torch.Tensor:
        """
        Get reward coefficients for a specific block by rolling the coefficient tensor.
        
        Args:
            block_idx: Block index (0 to num_blocks - 1)
            
        Returns:
            Reward coefficients for the specified block [num_envs]
        """
        return torch.roll(self.reward_coef, self.block_size * block_idx, dims=0)
    
    def augment_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Augment observations with exploration coefficient embeddings.
        
        Args:
            obs: Original observations [num_envs, obs_dim]
            
        Returns:
            Augmented observations [num_envs, obs_dim + embd_dim]
        """
        # Concatenate embeddings to observations
        return torch.cat([obs, self.coef_embd], dim=-1)
    
    @property
    def embd_dim(self) -> int:
        """Get embedding dimension."""
        return self.coef_embd.shape[-1]
    
    @property
    def num_blocks(self) -> int:
        """Get number of blocks."""
        return self.cfg.num_blocks

@configclass
class ExplorationCoefficientCfg(ModuleBaseCfg):
    """Configuration for exploration coefficient manager."""
    
    class_type: type[ExplorationCoefficient] = ExplorationCoefficient
    
    # Exploration type
    expl_type: Literal["mixed_expl_learn_param",  "mixed_expl_disjoint"] = "mixed_expl_learn_param"  # "mixed_expl_learn_param" | "mixed_expl_disjoint"
    
    # Block configuration
    num_blocks: int = MISSING  # Number of blocks
    # block_size will be computed automatically: num_envs // num_blocks
    
    # Embedding configuration
    embd_size: int = 1  # Embedding dimension (1 for learn_param mode)
    embd_init_range: tuple[float, float] = (50.0, 0.0)  # Embedding initialization range
    
    # Reward coefficient configuration
    reward_coef_type: str = "entropy"  # "entropy" | "none"
    reward_coef_scale: float = 0.005  # Reward coefficient scale
    reward_coef_range: tuple[float, float] = (0.5, 0.0)  # Reward coefficient range
