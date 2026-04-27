"""BaseEnvWorker — abstract sim-worker process body.

Per-task subclasses (e.g. `RoboTwinPlaceCupEnvWorker`) implement four methods:
  - `setup()`           : sapien/mujoco/etc. init (must run inside child)
  - `reset_envs(ids)`   : reset given env instances; return their initial ObsBatch
  - `step_envs(action)` : apply ActionBatch; return (next_obs, reward, done, info)
  - `teardown()`        : release sim resources

The provided `run_loop()` ties them together with the channels:
  put obs → wait for action → step → maybe emit trajectory → repeat.

Trajectory accumulation is intentionally minimal in this skeleton; Step 4 of
the plan (real RoboTwin run) wires the rollout buffer to `traj_ch`.
"""

from __future__ import annotations

import time
from dataclasses import field
from typing import Optional

from RoboRenForce.utils.configclass import configclass

from RRF_orchestra.protocol.channels import Channel
from RRF_orchestra.protocol.messages import ActionBatch, ObsBatch
from RRF_orchestra.workers.base_worker import BaseWorker, BaseWorkerCfg


@configclass
class BaseEnvWorkerCfg(BaseWorkerCfg):
    """Common knobs; subclasses add task-specific fields (seed, env_kwargs, ...)."""

    name: str = "env_worker"
    worker_id: int = 0
    num_envs_per_worker: int = 1
    chunk_size: int = 1                       # action chunk length
    obs_ch: Optional[Channel] = None          # this worker → InferenceWorker
    action_ch: Optional[Channel] = None       # InferenceWorker → this worker
    traj_ch: Optional[Channel] = None         # this worker → Learner (optional)
    # Block timeouts on the worker's hot path. Loose enough to absorb a slow
    # inference forward pass; tight enough to notice a stalled topology.
    obs_put_timeout_s: float = 30.0
    action_get_timeout_s: float = 30.0


class BaseEnvWorker(BaseWorker):
    """Abstract sim-worker. Subclass per (benchmark, task) pair."""

    cfg: BaseEnvWorkerCfg

    # ---- subclass MUST implement ------------------------------------------

    def reset_envs(self, env_ids: list[int]) -> ObsBatch:
        raise NotImplementedError

    def step_envs(self, action: ActionBatch):
        """Returns (next_obs: ObsBatch, reward, done, info_list)."""
        raise NotImplementedError

    # ---- provided ----------------------------------------------------------

    def run_loop(self) -> None:
        if self.cfg.obs_ch is None or self.cfg.action_ch is None:
            raise RuntimeError(
                f"{self.cfg.name}: obs_ch and action_ch must be wired before run()"
            )

        env_ids = list(range(self.cfg.num_envs_per_worker))
        obs = self.reset_envs(env_ids)

        while not self.should_stop():
            if not self._put_obs(obs):
                return  # stop requested or topology stalled past obs_put_timeout
            action = self._wait_for_action()
            if action is None:
                return

            next_obs, reward, done, info = self.step_envs(action)
            # Trajectory emission is left to subclasses (Step 4 wires the buffer).
            self._on_step(next_obs, reward, done, info)
            obs = next_obs

    def _put_obs(self, obs: ObsBatch) -> bool:
        """Put obs on the channel, polling for stop every `poll_interval_s`."""
        deadline = time.monotonic() + self.cfg.obs_put_timeout_s
        while not self.should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                self.cfg.obs_ch.put(
                    obs, timeout=min(self.cfg.poll_interval_s, remaining)
                )
                return True
            except Exception:
                continue
        return False

    def _wait_for_action(self) -> Optional[ActionBatch]:
        """Block for an action, re-checking stop every `poll_interval_s`."""
        deadline = time.monotonic() + self.cfg.action_get_timeout_s
        while not self.should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                return self.cfg.action_ch.get(
                    timeout=min(self.cfg.poll_interval_s, remaining)
                )
            except Exception:
                continue
        return None

    # ---- hooks subclasses may override ------------------------------------

    def _on_step(self, next_obs, reward, done, info) -> None:
        """Override to accumulate / emit trajectories. No-op by default."""
