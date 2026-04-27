"""Round-trip tests for the wire protocol.

Each cross-process test spawns a child via `mp.Process` to verify pickling and
shared-memory zero-copy semantics, not just same-process correctness.
"""

from __future__ import annotations

import time

import pytest
import torch
import torch.multiprocessing as mp

from RRF_distributed.protocol.channels import (
    ActionChannel,
    ControlChannel,
    ObsChannel,
    TrajChannel,
)
from RRF_distributed.protocol.messages import (
    PROTOCOL_VERSION,
    ActionBatch,
    ControlMsg,
    ObsBatch,
    Trajectory,
)
from RRF_distributed.protocol.shared_tensor import SharedTensorRef


# ----- SharedTensorRef ------------------------------------------------------


def test_shared_tensor_round_trip_same_process():
    t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    ref = SharedTensorRef.from_tensor(t)
    out = ref.materialize()
    assert out.shape == t.shape
    assert out.dtype == t.dtype
    assert torch.equal(out, t)


def test_shared_tensor_zero_copy_in_process():
    """In the same process, materialize() must point to the same storage."""
    t = torch.zeros(8, dtype=torch.uint8)
    t.fill_(7)
    ref = SharedTensorRef.from_tensor(t)
    out = ref.materialize()
    # Mutating one must mutate the other (zero-copy semantics).
    out[0] = 42
    assert t[0].item() == 42, "SharedTensorRef.materialize() did not return a shared view"


def test_shared_tensor_rejects_gpu_tensor():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    t = torch.zeros(4, device="cuda")
    with pytest.raises(ValueError, match="CPU"):
        SharedTensorRef.from_tensor(t)


def _child_materialize_and_modify(ref: SharedTensorRef, sentinel_q: mp.Queue):
    out = ref.materialize()
    sentinel_q.put(("shape", tuple(out.shape)))
    sentinel_q.put(("sum", float(out.sum().item())))
    out[0] = 99  # mutate in place — visible to parent if shared.
    sentinel_q.put(("done", True))


def test_shared_tensor_cross_process_zero_copy():
    """Spawn a child, hand it a SharedTensorRef, verify mutations propagate."""
    ctx = mp.get_context("spawn")
    t = torch.full((6,), 3, dtype=torch.uint8)
    ref = SharedTensorRef.from_tensor(t)

    sentinel: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_child_materialize_and_modify, args=(ref, sentinel))
    p.start()
    p.join(timeout=15)
    assert p.exitcode == 0, f"child failed (exitcode={p.exitcode})"

    msgs = {}
    while not sentinel.empty():
        k, v = sentinel.get_nowait()
        msgs[k] = v
    assert msgs["shape"] == (6,)
    assert msgs["sum"] == 18.0  # 6 * 3
    # Child mutated index 0 to 99 — parent's tensor must see it.
    assert t[0].item() == 99, "Cross-process mutation did not propagate; storage was copied"


# ----- Channels -------------------------------------------------------------


def _child_obs_producer(ch, n: int):
    for i in range(n):
        states = SharedTensorRef.from_tensor(torch.full((4,), float(i)))
        ch.put(ObsBatch(
            worker_ids=[0], env_ids=[i], step_ids=[i],
            states=states, timestamp=time.time(),
        ))
    # Keep feeder thread alive long enough to flush all queued items.
    time.sleep(2.0)


def test_obs_channel_cross_process_round_trip():
    ctx = mp.get_context("spawn")
    ch = ObsChannel(maxsize=8)
    p = ctx.Process(target=_child_obs_producer, args=(ch, 5))
    p.start()
    received = [ch.get(timeout=15) for _ in range(5)]
    p.join(timeout=10)
    assert p.exitcode == 0

    assert [m.env_ids[0] for m in received] == [0, 1, 2, 3, 4]
    for i, m in enumerate(received):
        t = m.states.materialize()
        assert torch.equal(t, torch.full((4,), float(i)))


def test_get_batch_drains_to_max():
    ctx = mp.get_context("spawn")
    ch = ObsChannel(maxsize=16)
    p = ctx.Process(target=_child_obs_producer, args=(ch, 5))
    p.start()
    # Drain BEFORE joining: shared-memory file descriptors are owned by the
    # producer's resource tracker, so we must materialize before it exits.
    batch = ch.get_batch(max_batch=8, timeout_s=10.0)
    p.join(timeout=10)
    assert p.exitcode == 0
    assert len(batch) == 5
    assert [b.env_ids[0] for b in batch] == [0, 1, 2, 3, 4]
    for i, b in enumerate(batch):
        assert torch.equal(b.states.materialize(), torch.full((4,), float(i)))


def test_get_batch_returns_empty_on_deadline():
    ch = ObsChannel(maxsize=4)
    started = time.monotonic()
    batch = ch.get_batch(max_batch=4, timeout_s=0.2)
    elapsed = time.monotonic() - started
    assert batch == []
    assert 0.15 < elapsed < 1.0  # waited roughly the deadline, not forever


def test_channel_type_check_rejects_wrong_message():
    ch = ObsChannel(maxsize=4)
    with pytest.raises(TypeError):
        ch.put(ControlMsg(kind="start", sender="x"))  # type: ignore[arg-type]


# ----- Message defaults -----------------------------------------------------


def test_message_protocol_version_defaults():
    for cls in (ObsBatch, ActionBatch, Trajectory, ControlMsg):
        m = cls()
        assert m.schema_version == PROTOCOL_VERSION


def test_message_constructs_with_minimal_fields():
    obs = ObsBatch(worker_ids=[0], env_ids=[0], step_ids=[0])
    assert obs.images is None and obs.states is None
    act = ActionBatch(worker_ids=[0], env_ids=[0], step_ids=[0],
                      actions=SharedTensorRef.from_tensor(torch.zeros(1, 4, 7)))
    assert act.weight_version == 0
    traj = Trajectory(worker_id=0, env_id=0)
    assert traj.weight_version_range == (0, 0)


def _child_traj_producer(ch):
    ref = SharedTensorRef.from_tensor(torch.arange(8, dtype=torch.float32))
    ch.put(Trajectory(worker_id=1, env_id=2, rewards=ref,
                      weight_version_range=(3, 5),
                      info={"return": 1.5}))
    time.sleep(2.0)


def test_traj_channel_cross_process():
    ctx = mp.get_context("spawn")
    ch = TrajChannel(maxsize=4)
    p = ctx.Process(target=_child_traj_producer, args=(ch,))
    p.start()
    msg = ch.get(timeout=15)
    p.join(timeout=10)
    assert p.exitcode == 0
    assert msg.worker_id == 1 and msg.env_id == 2
    assert msg.weight_version_range == (3, 5)
    assert msg.info == {"return": 1.5}
    assert torch.equal(msg.rewards.materialize(), torch.arange(8, dtype=torch.float32))


def _child_action_producer(ch):
    a = SharedTensorRef.from_tensor(torch.ones(2, 4, 7))
    ch.put(ActionBatch(worker_ids=[0, 1], env_ids=[0, 0], step_ids=[3, 4],
                       actions=a, weight_version=11))
    time.sleep(2.0)


def test_action_channel_round_trip():
    ctx = mp.get_context("spawn")
    ch = ActionChannel(maxsize=4)
    p = ctx.Process(target=_child_action_producer, args=(ch,))
    p.start()
    msg = ch.get(timeout=15)
    p.join(timeout=10)
    assert p.exitcode == 0
    assert msg.weight_version == 11
    out = msg.actions.materialize()
    assert out.shape == (2, 4, 7)
    assert torch.allclose(out, torch.ones(2, 4, 7))


def _child_control_producer(ch):
    ch.put(ControlMsg(kind="health_ping", sender="env_worker_2",
                      payload={"qsize": 7}))
    time.sleep(2.0)


def test_control_channel_round_trip():
    ctx = mp.get_context("spawn")
    ch = ControlChannel(maxsize=4)
    p = ctx.Process(target=_child_control_producer, args=(ch,))
    p.start()
    msg = ch.get(timeout=15)
    p.join(timeout=10)
    assert p.exitcode == 0
    assert msg.kind == "health_ping"
    assert msg.sender == "env_worker_2"
    assert msg.payload == {"qsize": 7}
