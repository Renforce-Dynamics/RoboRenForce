import math
from typing import Optional

import torch
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):
    """
    Trainable positional embeddings.

    Each position index in [0, max_len) has its own learnable embedding vector.
    """

    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, dim)

        Returns:
            Tensor of shape (batch, seq_len, dim) with positional embeddings added.
        """
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum supported length {self.max_len}."
            )
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        pos_emb = self.embedding(positions)  # (seq_len, dim)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, seq_len, self.dim)
        return x + pos_emb