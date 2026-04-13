import math
from typing import Optional

import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention block with batch-first API.

    Inputs:
        x: Tensor of shape (batch, seq_len, dim)
        attention_mask (optional): Tensor of shape (batch, seq_len, seq_len) or (1, seq_len, seq_len)
            - dtype=bool
            - True  means the position is visible.
            - False means the position is masked out.
        key_padding_mask (optional): Tensor of shape (batch, seq_len), dtype=bool
            - True  means the token is valid.
            - False means the token is padding and should not be attended to.

    Outputs:
        y: Tensor of shape (batch, seq_len, dim)
        attn_weights: Tensor of shape (batch, num_heads, seq_len, seq_len)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Thin wrapper around torch.nn.MultiheadAttention with batch-first inputs.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape

        # Prepare attention mask for torch.nn.MultiheadAttention
        attn_mask_arg: Optional[torch.Tensor] = None
        if attention_mask is not None:
            # Allow (1, L, L) or (L, L). Per-batch masks are not supported by nn.MultiheadAttention.
            if attention_mask.dim() == 3:
                if attention_mask.size(0) != 1:
                    raise ValueError(
                        "Per-batch attention_mask is not supported by MultiHeadSelfAttention "
                        "when using torch.nn.MultiheadAttention. "
                        "Please provide a shared mask with shape (1, L, L) or (L, L)."
                    )
                mask_2d = attention_mask[0]
            elif attention_mask.dim() == 2:
                mask_2d = attention_mask
            else:
                raise ValueError(
                    "attention_mask must have shape (batch, L, L), (1, L, L), or (L, L)."
                )

            # Our convention: True = visible, False = masked.
            # nn.MultiheadAttention expects attn_mask where True = masked.
            if mask_2d.dtype == torch.bool:
                attn_mask_arg = ~mask_2d
            else:
                attn_mask_arg = mask_2d

        # Prepare key padding mask: our convention True=valid, False=padding.
        key_padding_mask_arg: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (bsz, seq_len):
                raise ValueError(
                    f"key_padding_mask must have shape (batch, seq_len) = ({bsz}, {seq_len}), "
                    f"but got {tuple(key_padding_mask.shape)}."
                )
            if key_padding_mask.dtype == torch.bool:
                # nn.MultiheadAttention: True = padding (ignored)
                key_padding_mask_arg = ~key_padding_mask
            else:
                key_padding_mask_arg = key_padding_mask

        # nn.MultiheadAttention returns attention weights with shape:
        #   (batch, num_heads, seq_len, seq_len) when average_attn_weights=False and batch_first=True
        y, attn_weights = self.mha(
            x,
            x,
            x,
            attn_mask=attn_mask_arg,
            key_padding_mask=key_padding_mask_arg,
            need_weights=True,
            average_attn_weights=False,
        )

        return y, attn_weights