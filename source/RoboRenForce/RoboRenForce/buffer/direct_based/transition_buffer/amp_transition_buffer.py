from __future__ import annotations

import torch


class AMPTransitionBuffer:
    """Simple replay buffer for AMP / GAIL state transitions (s_t, s_{t+1})."""

    def __init__(self, obs_dim: int, capacity: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.capacity = capacity
        self.obs_dim = obs_dim

        self.obs = torch.zeros((capacity, obs_dim), device=self.device)
        self.next_obs = torch.zeros((capacity, obs_dim), device=self.device)
        self.ptr = 0
        self.is_full = False

    @torch.no_grad()
    def insert(self, obs: torch.Tensor, next_obs: torch.Tensor):
        """Insert a batch of transitions into the buffer."""
        batch_size = obs.shape[0]
        if obs.shape != next_obs.shape:
            raise ValueError(
                f"AMPTransitionBuffer.insert: obs and next_obs must have same shape, "
                f"got {obs.shape} and {next_obs.shape}."
            )

        if batch_size > self.capacity:
            # Keep only the last `capacity` samples
            obs = obs[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            batch_size = self.capacity

        end = self.ptr + batch_size
        if end <= self.capacity:
            self.obs[self.ptr:end] = obs
            self.next_obs[self.ptr:end] = next_obs
        else:
            first = self.capacity - self.ptr
            self.obs[self.ptr:] = obs[:first]
            self.next_obs[self.ptr:] = next_obs[:first]
            remain = batch_size - first
            if remain > 0:
                self.obs[:remain] = obs[first:]
                self.next_obs[:remain] = next_obs[first:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        if batch_size == self.capacity or self.ptr == 0:
            self.is_full = True

    def __len__(self) -> int:
        return self.capacity if self.is_full else self.ptr

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Yield mini-batches of (obs, next_obs) pairs."""
        total = len(self)
        if total == 0:
            return

        for _ in range(num_mini_batch):
            idx = torch.randint(0, total, (mini_batch_size,), device=self.device)
            yield self.obs[idx], self.next_obs[idx]


__all__ = ["AMPTransitionBuffer"]

