"""Type-checked thin wrappers over `torch.multiprocessing.Queue`.

Each channel carries exactly one message type. `get_batch(...)` is the
non-trivial method: InferenceWorker uses it to drain N obs (or wait until
deadline) so it can run a single batched forward pass.
"""

from __future__ import annotations

import queue
import time
from typing import Generic, Optional, TypeVar

import torch.multiprocessing as mp

from RRF_distributed.protocol.messages import (
    ActionBatch,
    ControlMsg,
    ObsBatch,
    Trajectory,
    WeightUpdate,
    assert_compatible,
)

M = TypeVar("M")


class Channel(Generic[M]):
    """A typed wrapper over `mp.Queue` with batched drain support.

    The channel does NOT own the underlying queue beyond instantiating it,
    so the same Channel object is the producer/consumer endpoint after
    being passed across process boundaries via pickle (mp handles that).
    """

    # Use the spawn context queue so that channels behave identically whether
    # the parent uses the default start method or "spawn". sapien/mujoco can
    # crash under fork; the Supervisor pins start_method to "spawn", so all
    # cross-process channels must be spawn-compatible.
    _ctx = mp.get_context("spawn")

    def __init__(self, msg_type: type, maxsize: int = 0):
        self._q = Channel._ctx.Queue(maxsize=maxsize)
        self._msg_type = msg_type

    def put(self, msg: M, timeout: Optional[float] = None) -> None:
        if not isinstance(msg, self._msg_type):
            raise TypeError(
                f"Channel expects {self._msg_type.__name__}, got {type(msg).__name__}"
            )
        self._q.put(msg, timeout=timeout)

    def put_nowait(self, msg: M) -> None:
        self.put(msg, timeout=0)

    def get(self, timeout: Optional[float] = None) -> M:
        msg = self._q.get(timeout=timeout)
        assert_compatible(msg)
        return msg

    def get_batch(self, max_batch: int, timeout_s: float) -> list[M]:
        """Drain up to `max_batch` items, returning as soon as one of:
          - we have `max_batch` items, OR
          - the wall-clock deadline expires AND we have ≥1 item, OR
          - the deadline expires with the queue empty (returns []).
        """
        deadline = time.monotonic() + timeout_s
        out: list[M] = []
        # Block until first item (or deadline).
        first_remaining = max(0.0, deadline - time.monotonic())
        try:
            first = self._q.get(timeout=first_remaining)
            assert_compatible(first)
            out.append(first)
        except queue.Empty:
            return out
        # Drain greedily until we hit max_batch or deadline.
        while len(out) < max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = self._q.get(timeout=remaining)
                assert_compatible(msg)
                out.append(msg)
            except queue.Empty:
                break
        return out

    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except NotImplementedError:
            # macOS doesn't implement qsize; callers must handle 0 conservatively.
            return 0

    def close(self) -> None:
        self._q.close()

    def __deepcopy__(self, memo):
        # Channels are reference-shared on purpose: producer and consumer must
        # see the same underlying queue. Deepcopy would also fail on the inner
        # mp.Queue (which refuses to be pickled outside of process spawn).
        return self


def ObsChannel(maxsize: int = 0) -> Channel[ObsBatch]:
    return Channel(ObsBatch, maxsize=maxsize)


def ActionChannel(maxsize: int = 0) -> Channel[ActionBatch]:
    return Channel(ActionBatch, maxsize=maxsize)


def TrajChannel(maxsize: int = 0) -> Channel[Trajectory]:
    return Channel(Trajectory, maxsize=maxsize)


def WeightChannel(maxsize: int = 0) -> Channel[WeightUpdate]:
    return Channel(WeightUpdate, maxsize=maxsize)


def ControlChannel(maxsize: int = 0) -> Channel[ControlMsg]:
    return Channel(ControlMsg, maxsize=maxsize)
