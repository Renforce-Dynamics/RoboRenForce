"""InferenceWorker — owns the VLA actor, serves batched inference.

Loop:
  1. Drain obs_ch via the configured batcher (Fixed or Dynamic).
  2. Stack the SharedTensorRefs into a single batched tensor.
  3. Run the policy forward (no_grad) → actions.
  4. Split per producing worker and send the slice down each `action_chs[wid]`.

The policy itself is opaque here: the orchestrator wires a `policy_factory`
callable that produces an object with `predict(obs_dict) -> dict` API. Step 3
of the plan introduces a real `policy_adapter.py` that wraps `VLAActor`; for
now we keep the interface narrow so the standalone test can use a noop.
"""

from __future__ import annotations

import time
from dataclasses import field
from typing import Any, Callable, Optional, Union

import torch

from RoboRenForce.utils.configclass import configclass

from RRF_orchestra.protocol.channels import Channel
from RRF_orchestra.protocol.messages import ActionBatch, ObsBatch
from RRF_orchestra.protocol.shared_tensor import SharedTensorRef
from RRF_orchestra.workers.base_worker import BaseWorker, BaseWorkerCfg
from RRF_orchestra.workers.batcher import DynamicBatcher, FixedBatcher

PolicyFactory = Callable[[], Any]
Batcher = Union[FixedBatcher, DynamicBatcher]


@configclass
class InferenceWorkerCfg(BaseWorkerCfg):
    name: str = "inference_worker"
    obs_ch: Optional[Channel] = None
    # action_chs: one channel per env-worker, indexed by worker_id.
    action_chs: list = field(default_factory=list)
    weight_ch: Optional[Channel] = None
    # Constructed in the child to avoid pickling torch modules.
    policy_factory: Optional[PolicyFactory] = None
    # Batching strategy. Default keeps latency bounded under variable load.
    batcher: Optional[Batcher] = None
    # If True, pin the policy to this device after construction.
    device: str = "cpu"


class InferenceWorker(BaseWorker):
    cfg: InferenceWorkerCfg

    def __init__(self, cfg: InferenceWorkerCfg):
        super().__init__(cfg)
        self._policy: Any = None
        self._batcher: Optional[Batcher] = None
        self._weight_version: int = 0

    def setup(self) -> None:
        if self.cfg.obs_ch is None:
            raise RuntimeError("InferenceWorker.cfg.obs_ch must be wired")
        if not self.cfg.action_chs:
            raise RuntimeError("InferenceWorker.cfg.action_chs must be wired")
        if self.cfg.policy_factory is None:
            raise RuntimeError("InferenceWorker.cfg.policy_factory must be set")
        self._policy = self.cfg.policy_factory()
        self._batcher = self.cfg.batcher or DynamicBatcher(
            max_batch=max(1, len(self.cfg.action_chs)),
            max_wait_ms=20.0,
        )

    def run_loop(self) -> None:
        while not self.should_stop():
            self._maybe_apply_weight_update()
            obs_batches = self._batcher.collect(self.cfg.obs_ch, self.should_stop)
            if not obs_batches:
                continue
            action_batches = self._infer(obs_batches)
            self._dispatch(action_batches)

    # ---- internals ---------------------------------------------------------

    def _maybe_apply_weight_update(self) -> None:
        ch = self.cfg.weight_ch
        if ch is None:
            return
        # Drain any pending weight updates without blocking; keep only newest.
        latest = None
        while True:
            try:
                latest = ch.get(timeout=0)
            except Exception:
                break
        if latest is None:
            return
        # Real swap is implemented in Step 3 (weight_sync.py). For now we just
        # bump the version stamp so produced ActionBatches reflect it.
        self._weight_version = getattr(latest, "weight_version", self._weight_version)

    def _infer(self, obs_batches: list[ObsBatch]) -> list[ActionBatch]:
        """Run a single batched forward pass over `obs_batches`."""
        flat_states, flat_owners = self._flatten(obs_batches)

        with torch.no_grad():
            actions_t = self._policy.predict({"states": flat_states})  # (B, chunk, A)
        if not isinstance(actions_t, torch.Tensor):
            raise TypeError(
                f"policy.predict must return a Tensor; got {type(actions_t).__name__}"
            )

        return self._unflatten_actions(actions_t, flat_owners, obs_batches)

    @staticmethod
    def _flatten(obs_batches: list[ObsBatch]):
        """Stack states across batches; remember which obs each row came from."""
        rows = []
        owners = []  # parallel list of (obs_index_in_list, row_index_in_obs)
        for oi, ob in enumerate(obs_batches):
            t = ob.states.materialize() if ob.states is not None else None
            if t is None:
                continue
            for ri in range(t.shape[0]):
                rows.append(t[ri])
                owners.append((oi, ri))
        if not rows:
            raise RuntimeError("InferenceWorker received obs batches with no states")
        return torch.stack(rows, dim=0), owners

    def _unflatten_actions(
        self,
        actions_t: torch.Tensor,
        owners: list[tuple[int, int]],
        obs_batches: list[ObsBatch],
    ) -> list[ActionBatch]:
        """Slice the batched action tensor back into per-ObsBatch ActionBatches."""
        per_obs_rows: dict[int, list[int]] = {}
        for global_idx, (oi, _ri) in enumerate(owners):
            per_obs_rows.setdefault(oi, []).append(global_idx)

        out: list[ActionBatch] = []
        for oi, ob in enumerate(obs_batches):
            row_idxs = per_obs_rows.get(oi, [])
            if not row_idxs:
                continue
            sub = actions_t[row_idxs].contiguous()
            out.append(
                ActionBatch(
                    worker_ids=list(ob.worker_ids),
                    env_ids=list(ob.env_ids),
                    step_ids=list(ob.step_ids),
                    actions=SharedTensorRef.from_tensor(sub.cpu()),
                    weight_version=self._weight_version,
                    timestamp=time.time(),
                )
            )
        return out

    def _dispatch(self, action_batches: list[ActionBatch]) -> None:
        """Route each ActionBatch to its producing env-worker's action channel."""
        n = len(self.cfg.action_chs)
        for ab in action_batches:
            if not ab.worker_ids:
                continue
            wid = ab.worker_ids[0]
            if not 0 <= wid < n:
                raise IndexError(
                    f"ActionBatch references worker_id={wid} but only "
                    f"{n} action channels are wired"
                )
            self.cfg.action_chs[wid].put(ab)
