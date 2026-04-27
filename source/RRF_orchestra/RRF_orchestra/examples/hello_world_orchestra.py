"""Hello-world for the orchestra runner — no real env, no real policy.

Run:

    python -m RRF_orchestra.examples.hello_world_orchestra

What it does:
  1. Builds a `TopologyCfg` with 2 fake env workers + 1 inference worker.
  2. Uses a noop `nn.Linear(4, 7)` policy and a fake algorithm that just
     bumps a counter so we can see metrics flow.
  3. Runs `OrchestraVLARunner.learn()` for 5 iterations.

This is the smallest end-to-end demonstration that env / inference / learner
processes are wired correctly. Adapt by swapping `_TrajEmittingEnvWorkerCfg`
for a real `BaseEnvWorker` subclass (e.g. `RoboTwinPlaceCupEnvWorkerCfg` from
`RRF_robotwin_vla_rl_tasks`) and the noop policy + algorithm for a real
`VLAActor` + `GRPOAlgorithm`.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass

from RRF_orchestra.adapters.algorithm_adapter import AlgorithmAdapter
from RRF_orchestra.orchestrator.orchestra_runner import (
    OrchestraVLARunner,
    OrchestraVLARunnerCfg,
)
from RRF_orchestra.orchestrator.topology import TopologyCfg
from RRF_orchestra.protocol.messages import ObsBatch, Trajectory
from RRF_orchestra.protocol.shared_tensor import SharedTensorRef
from RRF_orchestra.workers.env_worker import BaseEnvWorker, BaseEnvWorkerCfg
from RRF_orchestra.workers.inference_worker import InferenceWorkerCfg


# ----- Fake env worker that emits Trajectory messages -----------------------


class HelloEnvWorker(BaseEnvWorker):
    def setup(self) -> None:
        self._step = 0

    def reset_envs(self, env_ids):
        return self._make_obs(env_ids)

    def step_envs(self, action):
        b = self.cfg.num_envs_per_worker
        rewards = torch.full((b,), 1.0, dtype=torch.float32)
        log_probs = torch.full((b,), -0.5, dtype=torch.float32)
        if self.cfg.traj_ch is not None:
            traj = Trajectory(
                worker_id=self.cfg.worker_id,
                env_id=0,
                rewards=SharedTensorRef.from_tensor(rewards),
                log_probs=SharedTensorRef.from_tensor(log_probs),
                weight_version_range=(action.weight_version, action.weight_version),
            )
            try:
                self.cfg.traj_ch.put(traj, timeout=2.0)
            except Exception:
                pass
        self._step += 1
        next_obs = self._make_obs(list(range(b)))
        reward = torch.zeros(b)
        done = torch.zeros(b, dtype=torch.bool)
        return next_obs, reward, done, []

    def _make_obs(self, env_ids):
        states = torch.full((len(env_ids), 4), float(self._step), dtype=torch.float32)
        return ObsBatch(
            worker_ids=[self.cfg.worker_id] * len(env_ids),
            env_ids=env_ids,
            step_ids=[self._step] * len(env_ids),
            states=SharedTensorRef.from_tensor(states),
            timestamp=time.time(),
        )


@configclass
class HelloEnvWorkerCfg(BaseEnvWorkerCfg):
    class_type: type = HelloEnvWorker
    name: str = "hello_env"


# ----- Noop policy + fake algorithm -----------------------------------------


class HelloPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 7, bias=False)

    def predict(self, obs_dict):
        states = obs_dict["states"]
        b = states.shape[0]
        return torch.zeros(b, 1, 7, dtype=torch.float32)


def _policy_factory():
    return HelloPolicy()


def _optimizer_factory(policy):
    return torch.optim.SGD(policy.parameters(), lr=1e-4)


class HelloAlgorithm:
    def __init__(self):
        self.calls = 0

    def update_from_trajectories(self, trajectories, optimizer):
        self.calls += 1
        for g in optimizer.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
        optimizer.step()
        optimizer.zero_grad()
        return {
            "iter_calls": self.calls,
            "num_trajs": len(trajectories),
        }


def _algorithm_factory() -> AlgorithmAdapter:
    return AlgorithmAdapter(HelloAlgorithm())


# ----- Main -----------------------------------------------------------------


def main() -> None:
    env_cfg = HelloEnvWorkerCfg(
        num_envs_per_worker=1,
        chunk_size=1,
        obs_put_timeout_s=10.0,
        action_get_timeout_s=10.0,
    )
    inf_cfg = InferenceWorkerCfg(policy_factory=_policy_factory)
    topology = TopologyCfg(
        num_env_workers=2,
        env_worker_cfg=env_cfg,
        inference_worker_cfg=inf_cfg,
    )
    runner_cfg = OrchestraVLARunnerCfg(
        topology=topology,
        learner_policy_factory=_policy_factory,
        optimizer_factory=_optimizer_factory,
        algorithm_factory=_algorithm_factory,
        max_iterations=5,
        batch_size=2,
        weight_sync_every=1,
        collect_timeout_s=20.0,
    )

    def _on_iter_end(it: int, metrics: dict) -> None:
        print(f"[iter {it:>2}] {metrics}")

    runner = OrchestraVLARunner(runner_cfg)
    metrics = runner.learn(on_iter_end=_on_iter_end)
    print(f"\nfinished — collected {len(metrics)} iteration(s)")


if __name__ == "__main__":
    main()
