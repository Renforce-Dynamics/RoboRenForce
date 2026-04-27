"""AlgorithmAdapter — bridges Trajectory messages into core algorithm `update()`.

Core algorithms (e.g. `GRPOAlgorithm`) operate on flat tensors:
`logprobs`, `old_logprobs`, `advantages`, optional `entropy` / `ref_logprobs`.

Trajectory messages from the Supervisor carry per-step rewards / actions /
old log-probs in shared memory. The adapter:

  1. Materializes each Trajectory's tensors (zero-copy).
  2. Concatenates across trajectories.
  3. Hands the policy a chance to recompute current log-probs (so the gradient
     flows). The recomputation hook is supplied by the caller because the
     log-prob signature depends on the policy's action head.
  4. Calls `algorithm.update(...)` with the recomputed tensors.

Kept deliberately small: the adapter assumes the algorithm exposes either
`compute_advantages` + `update` (GRPO/PPO style) OR a single `update_from_trajectories`
method that takes trajectories directly (custom algorithms can opt in).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from RRF_orchestra.protocol.messages import Trajectory


class AlgorithmAdapter:
    """Reduce a list of `Trajectory` to flat tensors and call `algorithm.update`."""

    def __init__(
        self,
        algorithm: Any,
        recompute_logprobs_fn: Optional[Callable[[list[Trajectory]], torch.Tensor]] = None,
    ):
        self._alg = algorithm
        self._recompute_logprobs_fn = recompute_logprobs_fn

    @property
    def algorithm(self) -> Any:
        return self._alg

    def update(
        self,
        trajectories: list[Trajectory],
        optimizer: torch.optim.Optimizer,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Convert trajectories → flat tensors and run one optimizer step.

        Returns the algorithm's metric dict (already scalar-ized).
        """
        if not trajectories:
            return {}

        # Custom algorithms can opt out of the default reduction.
        if hasattr(self._alg, "update_from_trajectories"):
            return self._alg.update_from_trajectories(trajectories, optimizer)

        rewards, old_logprobs = self._flatten(trajectories)
        if hasattr(self._alg, "compute_advantages"):
            advantages = self._alg.compute_advantages(rewards)
        else:
            advantages = rewards - rewards.mean()

        if self._recompute_logprobs_fn is None:
            # If the user did not provide a recompute hook, treat old_logprobs
            # as the current ones — this only makes sense for offline /
            # behaviour-cloning style updates and is intentionally explicit.
            new_logprobs = old_logprobs
        else:
            new_logprobs = self._recompute_logprobs_fn(trajectories)
            if new_logprobs.shape != old_logprobs.shape:
                raise RuntimeError(
                    f"recompute_logprobs_fn returned shape {tuple(new_logprobs.shape)} "
                    f"but old_logprobs is {tuple(old_logprobs.shape)}"
                )

        return self._alg.update(
            logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            optimizer=optimizer,
            ref_logprobs=ref_logprobs,
        )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _flatten(trajectories: list[Trajectory]) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate `rewards` and `log_probs` across trajectories.

        Trajectories without `log_probs` cannot be used for an on-policy update;
        we raise rather than silently substitute zeros.
        """
        rewards_parts: list[torch.Tensor] = []
        logprobs_parts: list[torch.Tensor] = []
        for tr in trajectories:
            if tr.rewards is None:
                raise RuntimeError(
                    f"Trajectory(worker_id={tr.worker_id}, env_id={tr.env_id}) "
                    "has no rewards"
                )
            if tr.log_probs is None:
                raise RuntimeError(
                    f"Trajectory(worker_id={tr.worker_id}, env_id={tr.env_id}) "
                    "has no log_probs; an on-policy algorithm requires them. "
                    "Either populate log_probs in the env worker or use a custom "
                    "algorithm that exposes update_from_trajectories()."
                )
            rewards_parts.append(tr.rewards.materialize().flatten())
            logprobs_parts.append(tr.log_probs.materialize().flatten())
        return (
            torch.cat(rewards_parts, dim=0),
            torch.cat(logprobs_parts, dim=0),
        )
