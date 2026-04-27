"""OrchestraVLARunner — drop-in learner that drives a spawned topology.

The runner owns:
  - the live policy + optimizer (lives in the orchestrator process)
  - the algorithm adapter (consumes Trajectory messages)
  - the Supervisor (spawns env / inference workers)

Loop:

    sup.start_all()
    for it in range(max_iterations):
        trajs = sup.collect_trajectories(min_count=batch_size)
        stats = adapter.update(trajs, optimizer, ref_logprobs=...)
        if it % weight_sync_every == 0:
            msg = pack_state_dict(policy.state_dict(), version=it)
            sup.broadcast_weights(msg)
        on_iter_end(it, stats)
    sup.stop_all()

The runner does NOT subclass `BaseRunner` because BaseRunner's `__init__`
demands an env. Orchestra-mode envs live in worker processes; the
orchestrator never sees one. We keep the cfg shape compatible (same
`max_iterations` / `experiment_name` fields) so existing logger plumbing
slots in.
"""

from __future__ import annotations

import time
from dataclasses import field
from typing import Callable, Optional

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template import ClassTemplateBase, ClassTemplateBaseCfg

from RRF_orchestra.adapters.algorithm_adapter import AlgorithmAdapter
from RRF_orchestra.orchestrator.supervisor import Supervisor, TopologyError
from RRF_orchestra.orchestrator.topology import TopologyCfg
from RRF_orchestra.orchestrator.weight_sync import pack_state_dict
from RRF_orchestra.protocol.messages import Trajectory


PolicyFactory = Callable[[], nn.Module]
OptimizerFactory = Callable[[nn.Module], torch.optim.Optimizer]


@configclass
class OrchestraVLARunnerCfg(ClassTemplateBaseCfg):
    """Drop-in replacement for the single-process VLA RL runner cfg."""

    class_type: type = None  # set after OrchestraVLARunner defined

    topology: TopologyCfg = field(default_factory=TopologyCfg)

    # Construction is deferred so the orchestrator process owns the model.
    learner_policy_factory: Optional[PolicyFactory] = None
    optimizer_factory: Optional[OptimizerFactory] = None

    # Wraps the core algorithm; MUST be supplied by the caller.
    algorithm_factory: Optional[Callable[[], AlgorithmAdapter]] = None

    max_iterations: int = 1
    batch_size: int = 1                       # min trajectories per update
    weight_sync_every: int = 1                # iterations between weight pushes
    collect_timeout_s: float = 60.0
    experiment_name: str = "orchestra_vla"


class OrchestraVLARunner(ClassTemplateBase):
    cfg: OrchestraVLARunnerCfg

    def __init__(self, cfg: OrchestraVLARunnerCfg):
        self.cfg = cfg
        self._policy: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._adapter: Optional[AlgorithmAdapter] = None
        self._iter: int = 0
        self._iter_metrics: list[dict] = []

    # ---- public API --------------------------------------------------------

    def learn(
        self,
        on_iter_end: Optional[Callable[[int, dict], None]] = None,
    ) -> list[dict]:
        """Run `max_iterations` of role-orchestrated RL.

        Returns the per-iteration metric dicts collected during training so
        callers (and tests) can assert on convergence trends.
        """
        self._build_learner_side()
        with Supervisor(self.cfg.topology) as sup:
            sup.start_all()
            try:
                # Push initial weights so the inference worker starts from
                # the same parameters the learner holds.
                self._push_weights(sup, version=0)

                for it in range(self.cfg.max_iterations):
                    self._iter = it
                    if sup.should_stop():
                        break

                    trajs = sup.collect_trajectories(
                        min_count=self.cfg.batch_size,
                        timeout_s=self.cfg.collect_timeout_s,
                    )
                    if not trajs:
                        # Exhausted budget without trajectories — surface as
                        # a topology error so callers can react.
                        raise TopologyError(
                            f"iter={it}: no trajectories collected within "
                            f"{self.cfg.collect_timeout_s}s"
                        )

                    metrics = self._adapter.update(
                        trajectories=trajs,
                        optimizer=self._optimizer,
                    )
                    metrics_record = {"iter": it, "num_trajs": len(trajs), **metrics}
                    self._iter_metrics.append(metrics_record)

                    if on_iter_end is not None:
                        on_iter_end(it, metrics_record)

                    if (it + 1) % self.cfg.weight_sync_every == 0:
                        self._push_weights(sup, version=it + 1)

            finally:
                sup.stop_all()

        return self._iter_metrics

    # ---- internals ---------------------------------------------------------

    def _build_learner_side(self) -> None:
        if self.cfg.learner_policy_factory is None:
            raise ValueError("OrchestraVLARunnerCfg.learner_policy_factory is required")
        if self.cfg.optimizer_factory is None:
            raise ValueError("OrchestraVLARunnerCfg.optimizer_factory is required")
        if self.cfg.algorithm_factory is None:
            raise ValueError("OrchestraVLARunnerCfg.algorithm_factory is required")

        self._policy = self.cfg.learner_policy_factory()
        self._optimizer = self.cfg.optimizer_factory(self._policy)
        self._adapter = self.cfg.algorithm_factory()
        if not isinstance(self._adapter, AlgorithmAdapter):
            raise TypeError(
                f"algorithm_factory must return AlgorithmAdapter, "
                f"got {type(self._adapter).__name__}"
            )

    def _push_weights(self, sup: Supervisor, version: int) -> None:
        if self._policy is None:
            return
        if not self.cfg.topology.enable_weight_channel:
            return
        msg = pack_state_dict(self._policy.state_dict(), version=version)
        sup.broadcast_weights(msg)


OrchestraVLARunnerCfg.class_type = OrchestraVLARunner
