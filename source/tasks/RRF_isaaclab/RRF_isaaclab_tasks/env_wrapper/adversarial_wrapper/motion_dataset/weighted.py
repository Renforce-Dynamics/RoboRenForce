from __future__ import annotations

from typing import List

import torch

from RoboRenForce import configclass

from .base import MotionDataset, MotionDatasetCfg


class WeightedMotionDataset(MotionDataset):
    """Extension of MotionDataset with weighted transition sampling."""

    def __init__(
        self,
        cfg: MotionDatasetCfg,
        env,
        device: str = "cpu",
        traj_weights: List[float] | None = None,
        transition_weights: torch.Tensor | None = None,
    ):
        super().__init__(cfg, env, device)
        num_transitions = len(self.index_t)
        if transition_weights is not None:
            if transition_weights.shape[0] != num_transitions:
                raise ValueError(
                    "transition_weights must have length equal to number of transitions."
                )
            self.weights = transition_weights.to(device).clone()
        else:
            self.weights = torch.ones(len(self.index_t), device=device)

        self._traj_weights = traj_weights
        self.norm_weights()

    def norm_weights(self):
        """Normalize sampling weights to form a valid probability distribution."""
        self.weights = self.weights / (self.weights.sum() + 1e-9)

    def update_weights(
        self,
        weights: torch.Tensor,
        method: str = "sum",
        inplace: bool = True,
    ):
        """Update sampling weights using different aggregation methods."""
        if method in ["sum", "mean"]:
            self.weights += weights.to(self.weights.device)
            self.norm_weights()
        elif method == "replace":
            self.weights.copy_(weights.to(self.device))
            self.norm_weights()
        else:
            raise NotImplementedError(f"Update method {method} is not supported.")

    def _build_transition_weights_from_traj(self, traj_weights):
        """Convert trajectory-level weights to transition-level weights."""
        if traj_weights is None:
            return torch.ones(len(self.index_t))

        traj_weights = torch.tensor(traj_weights, dtype=torch.float32)

        weights = []
        for w, L in zip(traj_weights, self._traj_lengths):
            if L >= 2:
                weights.append(torch.full((L - 1,), float(w)))

        return torch.cat(weights, dim=0)

    def sample_batch(self, batch_size: int, replacement: bool = True):
        """Sample transition indices using current weights."""
        idx = torch.multinomial(self.weights, batch_size, replacement=replacement)
        t = self.index_t[idx]
        tp1 = self.index_tp1[idx]
        return t, tp1


@configclass
class WeightedMotionDatasetCfg(MotionDatasetCfg):
    """Configuration for weighted AMP MotionDataset."""

    class_type: type[WeightedMotionDataset] = WeightedMotionDataset


__all__ = ["WeightedMotionDataset", "WeightedMotionDatasetCfg"]

