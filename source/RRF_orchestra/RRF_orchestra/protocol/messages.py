"""Wire protocol — the single source of truth for cross-process payloads.

Every message carries `schema_version`. Workers refuse to start when their
PROTOCOL_VERSION differs from the orchestrator's. Bump PROTOCOL_VERSION on
any breaking change to a message shape.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Literal, Optional, Union

from RoboRenForce.utils.configclass import configclass

from RRF_orchestra.protocol.shared_tensor import SharedTensorRef

PROTOCOL_VERSION: int = 1


@configclass
class ObsBatch:
    """SimWorker → InferenceWorker. One batch of observations."""

    worker_ids: list[int] = field(default_factory=list)
    env_ids: list[int] = field(default_factory=list)
    step_ids: list[int] = field(default_factory=list)
    images: Optional[SharedTensorRef] = None       # uint8 (B, num_cams, C, H, W)
    states: Optional[SharedTensorRef] = None       # float32 (B, state_dim)
    languages: Union[list[str], SharedTensorRef, None] = None
    timestamp: float = 0.0
    schema_version: int = PROTOCOL_VERSION


@configclass
class ActionBatch:
    """InferenceWorker → SimWorker. Mirrors ObsBatch indexing."""

    worker_ids: list[int] = field(default_factory=list)
    env_ids: list[int] = field(default_factory=list)
    step_ids: list[int] = field(default_factory=list)
    actions: Optional[SharedTensorRef] = None      # float32 (B, chunk_size, action_dim)
    log_probs: Optional[SharedTensorRef] = None    # float32 (B,) — required for on-policy
    weight_version: int = 0
    timestamp: float = 0.0
    schema_version: int = PROTOCOL_VERSION


@configclass
class Trajectory:
    """SimWorker → Learner. One finished rollout segment of length T."""

    worker_id: int = 0
    env_id: int = 0
    obs: list = field(default_factory=list)         # list[ObsBatch] of length T
    actions: Optional[SharedTensorRef] = None       # (T, chunk_size, action_dim)
    rewards: Optional[SharedTensorRef] = None       # (T,)
    dones: Optional[SharedTensorRef] = None         # (T,) bool
    values: Optional[SharedTensorRef] = None        # (T,) — if critic available
    log_probs: Optional[SharedTensorRef] = None     # (T,)
    info: dict = field(default_factory=dict)
    weight_version_range: tuple[int, int] = (0, 0)  # min, max version used
    schema_version: int = PROTOCOL_VERSION


@configclass
class WeightUpdate:
    """Learner → InferenceWorker. After each gradient step (or every K steps)."""

    weight_version: int = 0
    state_dict_handles: dict = field(default_factory=dict)  # {param_name: SharedTensorRef}
    is_full: bool = True                             # False = LoRA delta only
    timestamp: float = 0.0
    schema_version: int = PROTOCOL_VERSION


CtrlKind = Literal["start", "stop", "pause", "resume",
                   "health_ping", "health_pong", "fatal_error"]


@configclass
class ControlMsg:
    """Bidirectional. Lifecycle and supervision."""

    kind: CtrlKind = "start"
    sender: str = ""
    payload: Optional[dict] = None
    timestamp: float = 0.0
    schema_version: int = PROTOCOL_VERSION


def assert_compatible(msg: Any) -> None:
    """Raise if a received message was produced by an incompatible protocol."""
    received = getattr(msg, "schema_version", None)
    if received != PROTOCOL_VERSION:
        raise RuntimeError(
            f"Protocol version mismatch: message {type(msg).__name__} carries "
            f"schema_version={received}, but this process is on PROTOCOL_VERSION="
            f"{PROTOCOL_VERSION}. Restart all workers from the same revision."
        )
