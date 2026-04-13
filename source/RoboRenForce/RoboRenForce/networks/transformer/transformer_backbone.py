from __future__ import annotations

"""
Standard, modular Transformer implementation.

This module provides a clean, reusable Transformer backbone that can be used in:
    - Policy networks (e.g., sequence of observations / actions)
    - Generative models (e.g., DiT-style token or latent transformers)
    - Sequence prediction models for system dynamics

Design goals:
    - Batch-first API: inputs are shaped as (batch, seq_len, dim)
    - Clear separation of components (MHA, FFN, block, positional encoding, backbone)
    - Explicit and well-documented masking behavior
    - Minimal assumptions about the semantics of tokens
"""

from typing import Optional, Literal

import torch
import torch.nn as nn

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBaseCfg
from .structures import (
    build_causal_mask,
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    TransformerBlock,
)


class TransformerBackbone(nn.Module):
    """
    Generic Transformer backbone.

    This module only processes token sequences and does not include any task-specific heads.
    It is intended to be reused for:
        - Policy networks (attach an action/value head on top)
        - DiT-style generative models (attach diffusion heads and time conditioning)
        - Sequence prediction / system dynamics (attach prediction heads for next state, reward, etc.)

    Input / output:
        - Inputs are token embeddings of shape (batch, seq_len, dim).
        - Outputs are transformed token embeddings of the same shape.
    """

    def __init__(self, cfg: "TransformerBackboneCfg"):
        """
        Args:
            cfg: Configuration object controlling all architectural hyper-parameters
                except the input embedding projection, which should be handled by
                the caller (e.g., policy encoder, tokenizer, etc.).
        """
        super().__init__()
        self.cfg = cfg

        # Core dimensions and topology
        self.dim = cfg.dim
        self.num_layers = cfg.num_layers
        self.causal = cfg.causal
        self.max_seq_len = cfg.max_seq_len
        self.use_final_layer_norm = cfg.use_final_layer_norm

        # Positional encoding module
        if cfg.positional_encoding == "none":
            self.pos_encoding = None
        elif cfg.positional_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEmbedding(
                dim=self.dim,
                max_len=self.max_seq_len,
            )
        elif cfg.positional_encoding == "learned":
            self.pos_encoding = LearnedPositionalEmbedding(
                dim=self.dim,
                max_len=self.max_seq_len,
            )
        else:
            raise ValueError(f"Unsupported positional_encoding: {cfg.positional_encoding}")

        # Optional time-step embedding for DiT-style conditioning
        if cfg.time_embed_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(cfg.time_embed_dim, self.dim),
                nn.SiLU(),
                nn.Linear(self.dim, self.dim),
            )
        else:
            self.time_mlp = None

        # Optional global conditioning embedding (e.g., task embedding)
        if cfg.cond_embed_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cfg.cond_embed_dim, self.dim),
                nn.SiLU(),
                nn.Linear(self.dim, self.dim),
            )
        else:
            self.cond_mlp = None

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(self.dim) if self.use_final_layer_norm else None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        time_embedding: Optional[torch.Tensor] = None,
        global_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens of shape (batch, seq_len, dim).
            attention_mask: Optional attention mask of shape (batch, seq_len, seq_len) or (1, seq_len, seq_len).
                - dtype=bool, True means visible, False means masked.
                - If `causal=True` and this argument is None, a standard causal mask is used.
            key_padding_mask: Optional key padding mask of shape (batch, seq_len), dtype=bool.
                - True means token is valid, False means it is padding.
            time_embedding: Optional time-step embedding, shape (batch, time_embed_dim).
                - If provided and `time_mlp` is configured, its projection is added to all tokens.
            global_condition: Optional global conditioning vector, shape (batch, cond_embed_dim).
                - If provided and `cond_mlp` is configured, its projection is added to all tokens.

        Returns:
            Tensor of shape (batch, seq_len, dim).
        """
        bsz, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(
                f"Input dim ({dim}) does not match model dim ({self.dim}). "
                "You should project inputs to the model dimension before calling this backbone."
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum supported length {self.max_seq_len}."
            )

        # Optional positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Optional time-step conditioning (e.g., diffusion time)
        if self.time_mlp is not None and time_embedding is not None:
            if time_embedding.dim() != 2 or time_embedding.shape[0] != bsz:
                raise ValueError(
                    "time_embedding must have shape (batch, time_embed_dim) when provided."
                )
            t_proj = self.time_mlp(time_embedding)  # (B, dim)
            x = x + t_proj.unsqueeze(1)  # broadcast over sequence length

        # Optional global conditioning (e.g., task id or latent)
        if self.cond_mlp is not None and global_condition is not None:
            if global_condition.dim() != 2 or global_condition.shape[0] != bsz:
                raise ValueError(
                    "global_condition must have shape (batch, cond_embed_dim) when provided."
                )
            c_proj = self.cond_mlp(global_condition)  # (B, dim)
            x = x + c_proj.unsqueeze(1)

        # Default causal mask if requested and not provided
        if self.causal and attention_mask is None:
            attention_mask = build_causal_mask(seq_len, device=x.device)

        # Pass through Transformer blocks
        for block in self.blocks:
            x, _ = block(
                x,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
            )

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x


@configclass
class TransformerBackboneCfg(ModuleBaseCfg):
    """
    Configuration for the generic Transformer backbone.

    This configuration is intended to be reused across:
        - Policy networks
        - DiT-style generative models
        - Sequence prediction / system dynamics models
    """

    class_type: type[nn.Module] = TransformerBackbone

    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 1024
    positional_encoding: Literal["none", "sinusoidal", "learned"] = "learned"
    causal: bool = False
    use_final_layer_norm: bool = True

    # Optional conditioning dimensions; set to None if unused.
    time_embed_dim: Optional[int] = None
    cond_embed_dim: Optional[int] = None

