"""End-to-end smoke test for Step 3 of task3.

Drives `OrchestraVLARunner.learn()` against a topology of fake env workers
+ a noop inference policy. Verifies:

  - `Topology.build()` wires the channels and clones env_worker_cfg correctly.
  - `Supervisor` spawns + joins workers under the runner's context manager.
  - Trajectories produced by env workers reach the learner via `traj_ch`.
  - `WeightUpdate` packed by `pack_state_dict` survives the broadcast and is
    applied without a shape mismatch.
  - The runner returns one metric dict per iteration; no zombie processes.
  - Worker fatal errors propagate as `TopologyError` from the learner loop.

The fake env worker emits a Trajectory after every step so the learner can
consume them at the configured `batch_size`. The fake "algorithm" just bumps
a counter so we can assert metrics flow end-to-end.
"""

from __future__ import annotations

import time
from typing import Optional

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass

from RRF_orchestra.adapters.algorithm_adapter import AlgorithmAdapter
from RRF_orchestra.orchestrator.orchestra_runner import (
    OrchestraVLARunner,
    OrchestraVLARunnerCfg,
)
from RRF_orchestra.orchestrator.supervisor import TopologyError
from RRF_orchestra.orchestrator.topology import TopologyCfg
from RRF_orchestra.orchestrator.weight_sync import (
    apply_weight_update,
    pack_state_dict,
)
from RRF_orchestra.protocol.messages import ObsBatch, Trajectory
from RRF_orchestra.protocol.shared_tensor import SharedTensorRef
from RRF_orchestra.workers.env_worker import BaseEnvWorker, BaseEnvWorkerCfg
from RRF_orchestra.workers.inference_worker import InferenceWorkerCfg


# ----- Fake env worker that emits Trajectory messages -----------------------


class _TrajEmittingEnvWorker(BaseEnvWorker):
    """Each `step_envs` call also pushes a 1-step Trajectory onto traj_ch."""

    def setup(self) -> None:
        self._step = 0

    def reset_envs(self, env_ids):
        return self._make_obs(env_ids)

    def step_envs(self, action):
        # Build a minimal Trajectory carrying a single (reward, log_prob) pair.
        # Length is num_envs_per_worker so the adapter sees a non-empty batch.
        b = self.cfg.num_envs_per_worker
        rewards = torch.full((b,), 1.0, dtype=torch.float32)
        log_probs = torch.full((b,), -0.5, dtype=torch.float32)
        traj = Trajectory(
            worker_id=self.cfg.worker_id,
            env_id=0,
            obs=[],
            rewards=SharedTensorRef.from_tensor(rewards),
            log_probs=SharedTensorRef.from_tensor(log_probs),
            weight_version_range=(action.weight_version, action.weight_version),
        )
        if self.cfg.traj_ch is not None:
            try:
                self.cfg.traj_ch.put(traj, timeout=2.0)
            except Exception:
                pass

        self._step += 1
        next_obs = self._make_obs(list(range(b)))
        reward = torch.zeros(b)
        done = torch.zeros(b, dtype=torch.bool)
        info: list[dict] = []
        return next_obs, reward, done, info

    def _make_obs(self, env_ids: list[int]) -> ObsBatch:
        states = torch.full((len(env_ids), 4), float(self._step), dtype=torch.float32)
        return ObsBatch(
            worker_ids=[self.cfg.worker_id] * len(env_ids),
            env_ids=env_ids,
            step_ids=[self._step] * len(env_ids),
            states=SharedTensorRef.from_tensor(states),
            timestamp=time.time(),
        )


@configclass
class _TrajEmittingEnvWorkerCfg(BaseEnvWorkerCfg):
    class_type: type = _TrajEmittingEnvWorker
    name: str = "fake_env"


# ----- Fake learner-side pieces ---------------------------------------------


class _NoopPolicyModule(nn.Module):
    """nn.Module so `state_dict()` / `pack_state_dict` round-trip is exercised."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 7, bias=False)

    def predict(self, obs_dict):
        states = obs_dict["states"]
        b = states.shape[0]
        return torch.zeros(b, 1, 7, dtype=torch.float32)


def _make_noop_policy_module():
    return _NoopPolicyModule()


class _FakeAlgorithm:
    """Records each `update_from_trajectories` call so the test can assert."""

    def __init__(self):
        self.calls: int = 0

    def update_from_trajectories(self, trajectories, optimizer):
        self.calls += 1
        # Pretend to take an optimizer step so the optimizer reference is exercised.
        for g in optimizer.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
        optimizer.step()
        optimizer.zero_grad()
        return {"fake_loss": 0.123, "calls": self.calls, "num_trajs": len(trajectories)}


# ----- Module-level factories (must pickle under spawn) ---------------------


def _learner_policy_factory() -> nn.Module:
    return _NoopPolicyModule()


def _optimizer_factory(policy: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(policy.parameters(), lr=1e-4)


def _algorithm_factory() -> AlgorithmAdapter:
    return AlgorithmAdapter(_FakeAlgorithm())


# ----- Tests ----------------------------------------------------------------


def _make_runner_cfg(
    num_env_workers: int,
    max_iterations: int,
    batch_size: int,
    weight_sync_every: int = 1,
) -> OrchestraVLARunnerCfg:
    env_cfg = _TrajEmittingEnvWorkerCfg(
        num_envs_per_worker=1,
        chunk_size=1,
        obs_put_timeout_s=5.0,
        action_get_timeout_s=5.0,
    )
    inf_cfg = InferenceWorkerCfg(
        policy_factory=_make_noop_policy_module,
    )
    topology = TopologyCfg(
        num_env_workers=num_env_workers,
        env_worker_cfg=env_cfg,
        inference_worker_cfg=inf_cfg,
    )
    return OrchestraVLARunnerCfg(
        topology=topology,
        learner_policy_factory=_learner_policy_factory,
        optimizer_factory=_optimizer_factory,
        algorithm_factory=_algorithm_factory,
        max_iterations=max_iterations,
        batch_size=batch_size,
        weight_sync_every=weight_sync_every,
        collect_timeout_s=20.0,
    )


def test_orchestra_runner_round_trip_2_envs_3_iters():
    cfg = _make_runner_cfg(num_env_workers=2, max_iterations=3, batch_size=2)
    runner = OrchestraVLARunner(cfg)
    metrics = runner.learn()

    assert len(metrics) == 3, f"expected 3 metric dicts, got {len(metrics)}: {metrics}"
    for it, m in enumerate(metrics):
        assert m["iter"] == it
        assert m["num_trajs"] >= 2
        assert m["fake_loss"] == pytest.approx(0.123)
        assert m["calls"] == it + 1


def test_orchestra_runner_single_env_single_iter():
    """Smallest possible topology — one env worker, one iteration."""
    cfg = _make_runner_cfg(num_env_workers=1, max_iterations=1, batch_size=1)
    runner = OrchestraVLARunner(cfg)
    metrics = runner.learn()
    assert len(metrics) == 1
    assert metrics[0]["num_trajs"] >= 1


def test_pack_and_apply_weight_update_round_trip():
    """`pack_state_dict` + `apply_weight_update` is a closed-form unit test —
    no spawn required, but keeps the orchestrator suite self-contained."""
    src = _NoopPolicyModule()
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(0.7)

    msg = pack_state_dict(src.state_dict(), version=42)
    assert msg.weight_version == 42
    assert set(msg.state_dict_handles.keys()) == set(src.state_dict().keys())

    dst = _NoopPolicyModule()
    with torch.no_grad():
        for p in dst.parameters():
            p.fill_(0.0)

    n = apply_weight_update(dst, msg)
    assert n == len(src.state_dict())
    for (name_src, p_src), (name_dst, p_dst) in zip(
        src.state_dict().items(), dst.state_dict().items()
    ):
        assert name_src == name_dst
        assert torch.allclose(p_src, p_dst), f"{name_src} not synced"


# ----- Failure-handling smoke ----------------------------------------------


class _ExplodingEnvWorker(BaseEnvWorker):
    """Worker whose `setup()` raises so the supervisor reports fatal_error."""

    def setup(self) -> None:
        raise RuntimeError("boom from env setup")

    def reset_envs(self, env_ids):
        return ObsBatch()

    def step_envs(self, action):
        return ObsBatch(), torch.zeros(1), torch.zeros(1, dtype=torch.bool), []


@configclass
class _ExplodingEnvWorkerCfg(BaseEnvWorkerCfg):
    class_type: type = _ExplodingEnvWorker
    name: str = "exploding_env"


def test_orchestra_runner_propagates_worker_fatal_as_topology_error():
    env_cfg = _ExplodingEnvWorkerCfg(
        num_envs_per_worker=1,
        obs_put_timeout_s=2.0,
        action_get_timeout_s=2.0,
    )
    inf_cfg = InferenceWorkerCfg(policy_factory=_make_noop_policy_module)
    topology = TopologyCfg(
        num_env_workers=1,
        env_worker_cfg=env_cfg,
        inference_worker_cfg=inf_cfg,
    )
    cfg = OrchestraVLARunnerCfg(
        topology=topology,
        learner_policy_factory=_learner_policy_factory,
        optimizer_factory=_optimizer_factory,
        algorithm_factory=_algorithm_factory,
        max_iterations=2,
        batch_size=1,
        collect_timeout_s=5.0,
    )
    runner = OrchestraVLARunner(cfg)
    with pytest.raises(TopologyError):
        runner.learn()
