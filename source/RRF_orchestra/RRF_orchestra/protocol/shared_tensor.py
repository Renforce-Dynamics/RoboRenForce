"""Zero-copy tensor handle for cross-process transport.

A SharedTensorRef wraps a tensor that lives in shared memory. Producers call
`SharedTensorRef.from_tensor(t)` (which calls `t.share_memory_()`); consumers
get the ref back over an `mp.Queue` and call `.materialize()` to obtain a
view of the same underlying storage — no byte copy.

Only the small handle (storage metadata + shape/dtype) crosses the queue;
the bytes stay in /dev/shm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SharedTensorRef:
    """Pickle-safe handle to a tensor stored in shared memory.

    Use `from_tensor(t)` to publish a tensor and `materialize()` to receive it
    in another process. Both calls are O(1) — no byte copy.
    """

    storage_payload: Any
    shape: tuple[int, ...]
    dtype: torch.dtype

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "SharedTensorRef":
        """Publish a tensor for cross-process consumption.

        The producer must keep `t` alive until the consumer has materialized
        it; otherwise the shared storage is reclaimed.
        """
        if not t.is_contiguous():
            t = t.contiguous()
        if t.device.type != "cpu":
            raise ValueError(
                f"SharedTensorRef only supports CPU tensors (got device={t.device}). "
                "Move to CPU before sharing; the consumer can move to GPU."
            )
        t.share_memory_()
        # torch.multiprocessing exposes ForkingPickler hooks on tensors that
        # produce a pickle-safe rebuild closure; we round-trip through that
        # to obtain the cross-process payload.
        from torch.multiprocessing.reductions import reduce_tensor

        rebuild_fn, args = reduce_tensor(t)
        return cls(
            storage_payload=(rebuild_fn, args),
            shape=tuple(t.shape),
            dtype=t.dtype,
        )

    def materialize(self) -> torch.Tensor:
        """Reconstruct the tensor in the current process. Zero-copy."""
        rebuild_fn, args = self.storage_payload
        return rebuild_fn(*args)
