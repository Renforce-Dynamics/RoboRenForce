"""End-to-end test for worker skeletons (Step 2 of task3).

Spawns one fake EnvWorker + one InferenceWorker (noop policy) connected by
real channels and verifies:
  - obs flows env → inference → action → env (closed loop),
  - the topology shuts down cleanly when the supervisor sends ControlMsg("stop"),
  - the InferenceWorker returns ActionBatches matching the producing worker_id.
"""

from __future__ import annotations

import time
from dataclasses import field
from typing import Optional

import pytest
import torch
import torch.multiprocessing as mp

from RoboRenForce.utils.configclass import configclass

from RRF_distributed.protocol.channels import (
    ActionChannel,
    ControlChannel,
    ObsChannel,
)
from RRF_distributed.protocol.messages import ControlMsg, ObsBatch
from RRF_distributed.protocol.shared_tensor import SharedTensorRef
from RRF_distributed.workers.base_worker import BaseWorker, BaseWorkerCfg
from RRF_distributed.workers.env_worker import BaseEnvWorker, BaseEnvWorkerCfg
from RRF_distributed.workers.inference_worker import (
    InferenceWorker,
    InferenceWorkerCfg,
)
from RRF_distributed.workers.batcher import DynamicBatcher, FixedBatcher

CTX = mp.get_context("spawn")


# ----- FakeEnvWorker --------------------------------------------------------
# Module-level so it survives pickling under spawn.


class _FakeEnvWorker(BaseEnvWorker):
    """Generates incrementing-state obs and counts received actions on a queue."""

    # Attached by the parent process before spawn (bypasses configclass deepcopy,
    # which refuses raw mp.Queue). Pickling across spawn handles it correctly.
    sentinel_q: Optional[mp.Queue] = None

    def setup(self) -> None:
        self._step = 0
        self._sentinel: mp.Queue = self.sentinel_q
        assert self._sentinel is not None, "sentinel_q must be attached before spawn"

    def reset_envs(self, env_ids):
        return self._make_obs(env_ids)

    def step_envs(self, action):
        # Confirm action arrived with the matching shape/worker_id.
        a = action.actions.materialize()
        assert action.worker_ids[0] == self.cfg.worker_id
        self._sentinel.put(("action", self._step, a.shape[0]))
        self._step += 1
        next_obs = self._make_obs(list(range(self.cfg.num_envs_per_worker)))
        # Reward/done are placeholders for the skeleton test.
        reward = torch.zeros(self.cfg.num_envs_per_worker)
        done = torch.zeros(self.cfg.num_envs_per_worker, dtype=torch.bool)
        info: list[dict] = []
        return next_obs, reward, done, info

    def teardown(self):
        try:
            self._sentinel.put(("teardown", self._step, 0))
        except Exception:
            pass

    def _make_obs(self, env_ids: list[int]) -> ObsBatch:
        states = torch.full(
            (len(env_ids), 4), float(self._step), dtype=torch.float32
        )
        return ObsBatch(
            worker_ids=[self.cfg.worker_id] * len(env_ids),
            env_ids=env_ids,
            step_ids=[self._step] * len(env_ids),
            states=SharedTensorRef.from_tensor(states),
            timestamp=time.time(),
        )


@configclass
class _FakeEnvWorkerCfg(BaseEnvWorkerCfg):
    class_type: type = _FakeEnvWorker
    name: str = "fake_env"


# ----- Noop policy -----------------------------------------------------------


class _NoopPolicy:
    def __init__(self, action_dim: int = 7, chunk_size: int = 1):
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def predict(self, obs_dict):
        states = obs_dict["states"]
        b = states.shape[0]
        return torch.zeros(b, self.chunk_size, self.action_dim, dtype=torch.float32)


def _make_noop_policy():
    return _NoopPolicy()


def _bad_policy_factory():
    raise RuntimeError("policy boom")


# ----- Driver ---------------------------------------------------------------


class _Topology:
    """Bag of references the test must keep alive until the children have
    deserialized — otherwise the parent's mp.Queue SemLocks get unlinked
    before the spawn child can rebuild them.
    """

    def __init__(self, num_envs: int, batcher):
        self.obs_ch = ObsChannel(maxsize=8)
        self.action_ch_0 = ActionChannel(maxsize=8)
        self.env_ctrl = ControlChannel(maxsize=4)
        self.inf_ctrl = ControlChannel(maxsize=4)
        self.sup_in = ControlChannel(maxsize=8)
        self.sentinel: mp.Queue = CTX.Queue()

        env_cfg = _FakeEnvWorkerCfg(
            worker_id=0,
            num_envs_per_worker=num_envs,
            obs_ch=self.obs_ch,
            action_ch=self.action_ch_0,
            ctrl_ch_in=self.env_ctrl,
            ctrl_ch_out=self.sup_in,
            obs_put_timeout_s=5.0,
            action_get_timeout_s=5.0,
        )
        inf_cfg = InferenceWorkerCfg(
            obs_ch=self.obs_ch,
            action_chs=[self.action_ch_0],
            policy_factory=_make_noop_policy,
            batcher=batcher,
            ctrl_ch_in=self.inf_ctrl,
            ctrl_ch_out=self.sup_in,
        )

        self.env_worker = _FakeEnvWorker(env_cfg)
        self.env_worker.sentinel_q = self.sentinel
        self.inf_worker = InferenceWorker(inf_cfg)

        self.p_env = CTX.Process(target=self.env_worker.run, name="env-0")
        self.p_inf = CTX.Process(target=self.inf_worker.run, name="inf")


def _drain_actions(sentinel: mp.Queue, want: int, timeout_s: float) -> list[int]:
    """Wait for `want` ('action', step, batch_size) messages."""
    deadline = time.monotonic() + timeout_s
    seen: list[int] = []
    while len(seen) < want and time.monotonic() < deadline:
        try:
            kind, step, _bs = sentinel.get(timeout=0.5)
        except Exception:
            continue
        if kind == "action":
            seen.append(step)
    return seen


# ----- Tests ----------------------------------------------------------------


def _shutdown(topo: "_Topology") -> None:
    topo.env_ctrl.put(ControlMsg(kind="stop", sender="test"))
    topo.inf_ctrl.put(ControlMsg(kind="stop", sender="test"))
    topo.p_env.join(timeout=10)
    topo.p_inf.join(timeout=10)
    if topo.p_env.is_alive():
        topo.p_env.terminate()
    if topo.p_inf.is_alive():
        topo.p_inf.terminate()
    topo.p_env.join(timeout=5)
    topo.p_inf.join(timeout=5)


def test_round_trip_with_dynamic_batcher():
    topo = _Topology(num_envs=2, batcher=DynamicBatcher(max_batch=4, max_wait_ms=50.0))
    topo.p_env.start()
    topo.p_inf.start()
    try:
        steps = _drain_actions(topo.sentinel, want=5, timeout_s=20.0)
        assert len(steps) >= 5, f"only saw {len(steps)} actions in 20s"
        # Steps should be monotonically non-decreasing (and effectively 0,1,2,...).
        assert steps == sorted(steps)
        assert steps[0] == 0
    finally:
        _shutdown(topo)

    assert topo.p_env.exitcode == 0, f"env exited with {topo.p_env.exitcode}"
    assert topo.p_inf.exitcode == 0, f"inference exited with {topo.p_inf.exitcode}"


def test_round_trip_with_fixed_batcher():
    # FixedBatcher(1) means each obs triggers one forward pass — degenerate
    # but exercises the fixed code path.
    topo = _Topology(
        num_envs=1, batcher=FixedBatcher(batch_size=1, per_get_timeout_s=0.5)
    )
    topo.p_env.start()
    topo.p_inf.start()
    try:
        steps = _drain_actions(topo.sentinel, want=3, timeout_s=20.0)
        assert len(steps) >= 3
    finally:
        _shutdown(topo)

    assert topo.p_env.exitcode == 0
    assert topo.p_inf.exitcode == 0


def test_inference_worker_reports_fatal_on_bad_policy():
    """A policy that raises must surface a fatal_error ControlMsg."""
    obs_ch = ObsChannel(maxsize=4)
    action_ch_0 = ActionChannel(maxsize=4)
    sup_in = ControlChannel(maxsize=4)
    inf_ctrl = ControlChannel(maxsize=4)

    cfg = InferenceWorkerCfg(
        obs_ch=obs_ch,
        action_chs=[action_ch_0],
        policy_factory=_bad_policy_factory,
        ctrl_ch_in=inf_ctrl,
        ctrl_ch_out=sup_in,
    )
    worker = InferenceWorker(cfg)
    p = CTX.Process(target=worker.run)
    p.start()
    p.join(timeout=10)
    assert p.exitcode != 0, "worker should have died on policy boom"

    msg = sup_in.get(timeout=5)
    assert msg.kind == "fatal_error"
    assert "policy boom" in msg.payload["message"]
