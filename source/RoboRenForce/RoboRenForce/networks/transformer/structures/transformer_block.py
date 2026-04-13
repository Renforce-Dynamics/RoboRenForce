import math
from typing import Optional

import torch
import torch.nn as nn

from .muti_head_attention import MultiHeadSelfAttention
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block.

    Structure:
        x = x + MHA(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.ffn = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention block
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        x = residual + self.dropout(attn_out)

        # Feed-forward block
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout(ffn_out)

        return x, attn_weights