"""Topology — declares N env workers, 1 inference worker, channels in between.

`TopologyCfg` is the user-facing config. `Topology.build()` materializes the
channels and returns a fully-wired `BuiltTopology` that the Supervisor can
spawn. Building is split from spawning so that test code can introspect or
swap channels without forking processes.

Channel fan-out (matches plan §3.3):
  obs_ch        : N producers → 1 consumer  (maxsize = 4 × N)
  action_ch[i]  : 1 producer  → 1 consumer  (per env worker, maxsize 4)
  traj_ch       : N producers → 1 consumer  (maxsize = 2 × N)
  weight_ch     : 1 producer  → 1 consumer  (maxsize 2; newest wins)
  ctrl_in_ch[i] : Supervisor → worker i     (maxsize 8)
  ctrl_out_ch   : workers → Supervisor      (single shared queue, maxsize 8 × M)
"""

from __future__ import annotations

import copy
from dataclasses import field
from typing import Optional

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template import ClassTemplateBase, ClassTemplateBaseCfg

from RRF_orchestra.protocol.channels import (
    ActionChannel,
    Channel,
    ControlChannel,
    ObsChannel,
    TrajChannel,
    WeightChannel,
)
from RRF_orchestra.workers.env_worker import BaseEnvWorker, BaseEnvWorkerCfg
from RRF_orchestra.workers.inference_worker import (
    InferenceWorker,
    InferenceWorkerCfg,
)


@configclass
class TopologyCfg(ClassTemplateBaseCfg):
    """User-facing topology config.

    Set `env_worker_cfg` to a concrete subclass (e.g. `RoboTwinPlaceCupEnvWorkerCfg`).
    The same cfg is cloned `num_env_workers` times; only `worker_id` differs.
    """

    class_type: type = None  # set after Topology defined

    num_env_workers: int = 1
    env_worker_cfg: Optional[BaseEnvWorkerCfg] = None
    inference_worker_cfg: InferenceWorkerCfg = field(
        default_factory=InferenceWorkerCfg
    )
    health_timeout_s: float = 10.0
    # Whether to allocate a TrajChannel. Pure-inference smoke tests can skip it.
    enable_traj_channel: bool = True
    # Whether to allocate a WeightChannel. Tests with a frozen policy can skip.
    enable_weight_channel: bool = True


class BuiltTopology:
    """Wired channels + worker instances. Spawning is the Supervisor's job."""

    def __init__(
        self,
        env_workers: list[BaseEnvWorker],
        inference_worker: InferenceWorker,
        obs_ch: Channel,
        action_chs: list[Channel],
        traj_ch: Optional[Channel],
        weight_ch: Optional[Channel],
        ctrl_in_chs: list[Channel],
        ctrl_out_ch: Channel,
    ):
        self.env_workers = env_workers
        self.inference_worker = inference_worker
        self.obs_ch = obs_ch
        self.action_chs = action_chs
        self.traj_ch = traj_ch
        self.weight_ch = weight_ch
        self.ctrl_in_chs = ctrl_in_chs
        self.ctrl_out_ch = ctrl_out_ch

    @property
    def num_workers(self) -> int:
        # +1 for the inference worker
        return len(self.env_workers) + 1


class Topology(ClassTemplateBase):
    cfg: TopologyCfg

    def __init__(self, cfg: TopologyCfg):
        self.cfg = cfg

    def build(self) -> BuiltTopology:
        if self.cfg.env_worker_cfg is None:
            raise ValueError(
                "TopologyCfg.env_worker_cfg must be set to a concrete BaseEnvWorkerCfg"
            )
        n = self.cfg.num_env_workers
        if n <= 0:
            raise ValueError(f"num_env_workers must be ≥ 1, got {n}")

        # ---- channels --------------------------------------------------------
        obs_ch = ObsChannel(maxsize=max(4, 4 * n))
        action_chs = [ActionChannel(maxsize=4) for _ in range(n)]
        traj_ch = TrajChannel(maxsize=max(2, 2 * n)) if self.cfg.enable_traj_channel else None
        weight_ch = WeightChannel(maxsize=2) if self.cfg.enable_weight_channel else None

        # One ctrl-in channel per worker (Supervisor → worker), plus one shared
        # ctrl-out channel (worker → Supervisor). The +1 ctrl_in covers the
        # inference worker.
        ctrl_in_chs = [ControlChannel(maxsize=8) for _ in range(n + 1)]
        ctrl_out_ch = ControlChannel(maxsize=max(8, 8 * (n + 1)))

        # ---- env workers -----------------------------------------------------
        env_workers: list[BaseEnvWorker] = []
        for i in range(n):
            ecfg = copy.copy(self.cfg.env_worker_cfg)
            ecfg.worker_id = i
            ecfg.name = f"{ecfg.name or 'env'}-{i}"
            ecfg.obs_ch = obs_ch
            ecfg.action_ch = action_chs[i]
            ecfg.traj_ch = traj_ch
            ecfg.ctrl_ch_in = ctrl_in_chs[i]
            ecfg.ctrl_ch_out = ctrl_out_ch
            env_workers.append(ecfg.construct_from_cfg())

        # ---- inference worker ------------------------------------------------
        icfg = copy.copy(self.cfg.inference_worker_cfg)
        icfg.obs_ch = obs_ch
        icfg.action_chs = action_chs
        icfg.weight_ch = weight_ch
        icfg.ctrl_ch_in = ctrl_in_chs[n]
        icfg.ctrl_ch_out = ctrl_out_ch
        inference_worker = InferenceWorker(icfg)

        return BuiltTopology(
            env_workers=env_workers,
            inference_worker=inference_worker,
            obs_ch=obs_ch,
            action_chs=action_chs,
            traj_ch=traj_ch,
            weight_ch=weight_ch,
            ctrl_in_chs=ctrl_in_chs,
            ctrl_out_ch=ctrl_out_ch,
        )


TopologyCfg.class_type = Topology
