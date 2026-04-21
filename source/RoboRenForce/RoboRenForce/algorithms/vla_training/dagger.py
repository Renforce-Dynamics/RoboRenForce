"""
DAgger — Dataset Aggregation for online imitation learning of VLA policies.

DAgger iteratively collects data by running the current policy, then
trains on expert corrections (interventions). The key insight is that
we only store trajectory segments where the expert intervened, so the
policy learns from states it actually encounters.

Algorithm:
    1. Run current policy in environment
    2. Expert intervenes when policy fails (intervention flags)
    3. Store expert-intervention trajectories in replay buffer
    4. Sample from replay buffer and train with supervised loss (SFT)
    5. Repeat

Reference: RLinf rlinf/workers/actor/fsdp_dagger_policy_worker.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.prototype.embodied import BasePolicy, ForwardType


@dataclass
class Trajectory:
    """A single trajectory (or sub-trajectory) of experience.

    Stores obs, actions, and optional metadata for a contiguous
    sequence of timesteps.
    """
    obs: dict[str, torch.Tensor]    # {key: [T, ...]}
    actions: torch.Tensor           # [T, action_dim]
    is_expert: torch.Tensor         # [T] bool flags
    rewards: Optional[torch.Tensor] = None  # [T]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return self.actions.shape[0]

    @classmethod
    def extract_expert_segments(cls, traj: "Trajectory") -> Optional["Trajectory"]:
        """Extract sub-trajectory where expert intervened.

        Returns None if no expert interventions in this trajectory.
        """
        mask = traj.is_expert
        if not mask.any():
            return None

        expert_obs = {k: v[mask] for k, v in traj.obs.items()}
        expert_actions = traj.actions[mask]
        expert_flags = traj.is_expert[mask]
        expert_rewards = traj.rewards[mask] if traj.rewards is not None else None

        return cls(
            obs=expert_obs,
            actions=expert_actions,
            is_expert=expert_flags,
            rewards=expert_rewards,
            metadata=traj.metadata,
        )


class TrajectoryReplayBuffer:
    """Simple replay buffer for storing expert trajectories.

    Supports:
    - FIFO eviction when max capacity is reached
    - Uniform random sampling of individual transitions
    - Window-based sampling from recent trajectories
    """

    def __init__(
        self,
        max_trajectories: int = 1000,
        sample_window_size: int = 100,
    ):
        self.max_trajectories = max_trajectories
        self.sample_window_size = sample_window_size
        self._buffer: deque[Trajectory] = deque(maxlen=max_trajectories)
        self._total_transitions = 0

    def add(self, traj: Trajectory):
        """Add a trajectory to the buffer."""
        if len(self._buffer) == self.max_trajectories:
            evicted = self._buffer[0]
            self._total_transitions -= len(evicted)
        self._buffer.append(traj)
        self._total_transitions += len(traj)

    def add_from_rollout(self, traj: Trajectory):
        """Extract expert segments from a rollout and add to buffer."""
        expert_traj = Trajectory.extract_expert_segments(traj)
        if expert_traj is not None and len(expert_traj) > 0:
            self.add(expert_traj)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch of transitions uniformly from recent trajectories.

        Args:
            batch_size: number of transitions to sample

        Returns:
            dict with batched obs, actions
        """
        # Use recent window
        window = list(self._buffer)[-self.sample_window_size:]

        # Concatenate all transitions in window
        all_obs = {}
        all_actions = []
        for traj in window:
            for k, v in traj.obs.items():
                if k not in all_obs:
                    all_obs[k] = []
                all_obs[k].append(v)
            all_actions.append(traj.actions)

        cat_obs = {k: torch.cat(v, dim=0) for k, v in all_obs.items()}
        cat_actions = torch.cat(all_actions, dim=0)

        total = cat_actions.shape[0]
        if total == 0:
            raise RuntimeError("Replay buffer is empty")

        # Random sample
        indices = torch.randint(0, total, (min(batch_size, total),))
        batch_obs = {k: v[indices] for k, v in cat_obs.items()}
        batch_actions = cat_actions[indices]

        return {"obs": batch_obs, "actions": batch_actions}

    def is_ready(self, min_size: int = 1) -> bool:
        """Check if buffer has enough data for training."""
        return len(self._buffer) >= min_size

    @property
    def num_trajectories(self) -> int:
        return len(self._buffer)

    @property
    def total_transitions(self) -> int:
        return self._total_transitions


class DAggerAlgorithm(ModuleBase):
    """DAgger algorithm for online imitation learning.

    Usage:
        dagger = DAggerAlgorithm(cfg)

        # During rollout collection:
        dagger.receive_trajectory(rollout_traj)

        # During training:
        if dagger.is_ready():
            metrics = dagger.update(policy, optimizer)
    """

    def __init__(self, cfg: "DAggerAlgorithmCfg"):
        super().__init__()
        self.cfg = cfg
        self.replay_buffer = TrajectoryReplayBuffer(
            max_trajectories=cfg.max_trajectories,
            sample_window_size=cfg.sample_window_size,
        )
        self._step = 0

    def receive_trajectory(self, traj: Trajectory):
        """Receive a rollout trajectory and extract expert segments."""
        self.replay_buffer.add_from_rollout(traj)

    def receive_expert_data(self, traj: Trajectory):
        """Directly add expert demonstration data."""
        self.replay_buffer.add(traj)

    def is_ready(self) -> bool:
        """Check if enough data is available for training."""
        return self.replay_buffer.is_ready(self.cfg.min_buffer_size)

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        policy: BasePolicy,
    ) -> dict[str, torch.Tensor]:
        """Compute SFT loss on sampled batch.

        Args:
            batch: dict with "obs" (dict) and "actions" (tensor)
            policy: the policy to train

        Returns:
            dict with total_loss, action_loss
        """
        obs = batch["obs"]
        target_actions = batch["actions"]
        if target_actions.dim() == 2:
            target_actions = target_actions.unsqueeze(1)

        result = policy.forward(
            ForwardType.SFT,
            obs=obs,
            target_actions=target_actions,
        )

        if "loss" in result:
            action_loss = result["loss"]
        else:
            raise ValueError("Policy must return 'loss' from SFT forward")

        return {"total_loss": action_loss, "action_loss": action_loss.detach()}

    def update(
        self,
        policy: BasePolicy,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Sample from replay buffer and do one training step.

        Returns:
            dict with scalar metrics
        """
        batch = self.replay_buffer.sample(self.cfg.batch_size)
        loss_dict = self.compute_loss(batch, policy)

        total_loss = loss_dict["total_loss"]

        optimizer.zero_grad()
        total_loss.backward()
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(policy.trainable_parameters()),
                self.cfg.max_grad_norm,
            )
        optimizer.step()
        self._step += 1

        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }
        metrics["buffer/num_trajectories"] = self.replay_buffer.num_trajectories
        metrics["buffer/total_transitions"] = self.replay_buffer.total_transitions
        return metrics

    def update_epochs(
        self,
        policy: BasePolicy,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 1,
    ) -> dict[str, float]:
        """Run multiple update epochs (DAgger typically does 1).

        Returns:
            dict with averaged metrics
        """
        all_metrics = []
        for _ in range(num_epochs):
            metrics = self.update(policy, optimizer)
            all_metrics.append(metrics)

        avg_metrics = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            avg_metrics[key] = sum(vals) / len(vals)
        return avg_metrics


@configclass
class DAggerAlgorithmCfg(ModuleBaseCfg):
    """DAgger algorithm configuration."""

    class_type: type[DAggerAlgorithm] = DAggerAlgorithm

    # Replay buffer
    max_trajectories: int = 1000
    sample_window_size: int = 100
    min_buffer_size: int = 5            # min trajectories before training

    # Training
    batch_size: int = 64
    update_epoch: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
