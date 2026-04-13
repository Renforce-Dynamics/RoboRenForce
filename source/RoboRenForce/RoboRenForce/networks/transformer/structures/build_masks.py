import math
from typing import Optional

import torch
import torch.nn as nn


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build a standard causal (autoregressive) attention mask.

    Shape: (1, seq_len, seq_len), dtype=bool
        - True  means the position is visible.
        - False means the position is masked out.
    """
    mask = torch.tril(torch.ones((1, seq_len, seq_len), device=device, dtype=torch.bool))
    return mask