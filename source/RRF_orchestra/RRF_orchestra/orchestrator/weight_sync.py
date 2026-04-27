"""Weight sync between Learner and InferenceWorker via shared-memory tensors.

The Learner publishes its current `state_dict` as a `WeightUpdate` whose
`state_dict_handles` is `{name: SharedTensorRef}`. The InferenceWorker
materializes each ref and copies into the live policy in place — copies, not
references, because the live policy must keep its own storage so that the
next gradient step can mutate the Learner's tensors without disturbing
inference.

Two helpers:
  - `pack_state_dict(sd, version)` : Learner side
  - `apply_weight_update(policy, msg)` : InferenceWorker side

These are kept tiny and pure so that both can be unit-tested without spawning
processes.
"""

from __future__ import annotations

import time
from typing import Mapping

import torch
import torch.nn as nn

from RRF_orchestra.protocol.messages import WeightUpdate
from RRF_orchestra.protocol.shared_tensor import SharedTensorRef


def pack_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    version: int,
    is_full: bool = True,
) -> WeightUpdate:
    """Convert a `state_dict` into a `WeightUpdate` whose tensors live in shared mem.

    The caller MUST keep `state_dict` alive until the consumer has applied the
    update. Detaching here would copy; instead we move each tensor to CPU (if
    needed) and call `share_memory_()` in place via `SharedTensorRef.from_tensor`.
    """
    handles: dict[str, SharedTensorRef] = {}
    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"state_dict[{name!r}] must be a Tensor, got {type(t).__name__}"
            )
        cpu_t = t.detach().cpu().contiguous()
        handles[name] = SharedTensorRef.from_tensor(cpu_t)

    return WeightUpdate(
        weight_version=version,
        state_dict_handles=handles,
        is_full=is_full,
        timestamp=time.time(),
    )


@torch.no_grad()
def apply_weight_update(policy: nn.Module, msg: WeightUpdate) -> int:
    """Copy each tensor in `msg` into the live `policy` parameters / buffers.

    Returns the number of tensors successfully copied. Missing keys are
    skipped (LoRA / partial updates) but a hard mismatch on shape raises.
    """
    if not isinstance(msg, WeightUpdate):
        raise TypeError(f"expected WeightUpdate, got {type(msg).__name__}")

    target = dict(policy.state_dict())
    copied = 0
    for name, ref in msg.state_dict_handles.items():
        if name not in target:
            # Partial / LoRA updates may carry fewer keys; not an error.
            continue
        src = ref.materialize()
        dst = target[name]
        if dst.shape != src.shape:
            raise RuntimeError(
                f"weight_sync: shape mismatch for {name!r}: "
                f"target {tuple(dst.shape)} vs source {tuple(src.shape)}"
            )
        dst.copy_(src.to(dst.device, dtype=dst.dtype, non_blocking=False))
        copied += 1
    return copied
