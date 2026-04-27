"""Batchers used by InferenceWorker to coalesce ObsBatches before a forward pass.

Two strategies, picked per benchmark:
  - FixedBatcher: wait for *exactly* `batch_size` obs before returning. Use when
    you have N == batch_size sim workers and each one produces one obs/step,
    so the batch is naturally aligned.
  - DynamicBatcher: wait up to `max_wait_ms` for whatever is available, return
    as soon as either limit fires. Use under variable sim latencies.
"""

from __future__ import annotations

from RRF_orchestra.protocol.channels import Channel
from RRF_orchestra.protocol.messages import ObsBatch


class FixedBatcher:
    """Block until exactly `batch_size` items are drained (or stop_check fires)."""

    def __init__(self, batch_size: int, per_get_timeout_s: float = 1.0):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size
        self.per_get_timeout_s = per_get_timeout_s

    def collect(self, ch: Channel, stop_check) -> list[ObsBatch]:
        """Drain `batch_size` items. Returns fewer (possibly empty) only on stop."""
        out: list[ObsBatch] = []
        while len(out) < self.batch_size and not stop_check():
            batch = ch.get_batch(
                max_batch=self.batch_size - len(out),
                timeout_s=self.per_get_timeout_s,
            )
            out.extend(batch)
        return out


class DynamicBatcher:
    """Drain up to `max_batch` items, returning early after `max_wait_s`."""

    def __init__(self, max_batch: int, max_wait_ms: float):
        if max_batch <= 0:
            raise ValueError(f"max_batch must be positive, got {max_batch}")
        if max_wait_ms <= 0:
            raise ValueError(f"max_wait_ms must be positive, got {max_wait_ms}")
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000.0

    def collect(self, ch: Channel, stop_check) -> list[ObsBatch]:
        # `get_batch` already implements the deadline + max-batch contract.
        # `stop_check` is checked only via the deadline expiring.
        del stop_check  # no fine-grained interrupt; deadline keeps us responsive
        return ch.get_batch(max_batch=self.max_batch, timeout_s=self.max_wait_s)
