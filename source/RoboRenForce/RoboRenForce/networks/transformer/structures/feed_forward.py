import math
from typing import Optional

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used inside Transformer blocks.

    Standard structure:
        FFN(x) = Linear(dim -> hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim -> dim)
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x