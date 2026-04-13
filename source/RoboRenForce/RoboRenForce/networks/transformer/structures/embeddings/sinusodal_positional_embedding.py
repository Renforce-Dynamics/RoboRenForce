import math
from typing import Optional

import torch
import torch.nn as nn

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Non-trainable sinusoidal positional embeddings, as in the original Transformer paper.

    This module adds positional information to token embeddings based only on sequence index.
    """

    def __init__(self, dim: int, max_len: int = 10_000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (max_len, dim) -> (1, max_len, dim) for broadcasting over batch
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, dim)
        Returns:
            Tensor of shape (batch, seq_len, dim) with positional embeddings added.
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum supported length {self.max_len}."
            )
        return x + self.pe[:, :seq_len, :].to(x.dtype)