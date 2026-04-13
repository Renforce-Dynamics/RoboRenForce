from __future__ import annotations

import torch
from RoboRenForce import configclass
from dataclasses import MISSING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from RoboRenForce.buffer.replay_buffer_base import ReplayBufferBaseCfg


class FlowReplayBuffer:
    """Replay buffer for belief flow model training.
    
    Stores:
    - low_obs: Low-frequency observations
    - high_obs: High-frequency observations (may be None/zero for some timesteps)
    - action: Actions taken
    - belief_h: Belief state before flow prediction
    - belief_h_pred: Belief state after flow prediction
    - belief_h_new: Belief state after observation update
    """
    
    def __init__(
        self,
        low_obs_dim: int,
        high_obs_dim: int,
        action_dim: int,
        belief_dim: int,
        buffer_size: int = 100_000,
        device: str = "cpu",
    ):
        """Initialize the replay buffer.
        
        Args:
            low_obs_dim: Dimension of low-frequency observations
            high_obs_dim: Dimension of high-frequency observations
            action_dim: Dimension of actions
            belief_dim: Dimension of belief state
            buffer_size: Maximum number of transitions per environment
            device: Device to store the buffer on
        """
        self.device = device
        self.buffer_size = buffer_size
        
        self.low_obs_dim = low_obs_dim
        self.high_obs_dim = high_obs_dim
        self.action_dim = action_dim
        self.belief_dim = belief_dim
        
        # Buffer will be initialized lazily
        self.replay_bufs = None
        self.num_envs = None
        self.step = 0
        self.num_transitions = 0
    
    def _initialize_buffer(self, num_envs: int):
        """Initialize the replay buffers dynamically."""
        self.num_envs = num_envs
        
        self.replay_bufs = {
            "low_obs": torch.zeros(num_envs, self.buffer_size, self.low_obs_dim, device=self.device),
            "high_obs": torch.zeros(num_envs, self.buffer_size, self.high_obs_dim, device=self.device),
            "action": torch.zeros(num_envs, self.buffer_size, self.action_dim, device=self.device),
            "belief_h": torch.zeros(num_envs, self.buffer_size, self.belief_dim, device=self.device),
            "belief_h_pred": torch.zeros(num_envs, self.buffer_size, self.belief_dim, device=self.device),
            "belief_h_new": torch.zeros(num_envs, self.buffer_size, self.belief_dim, device=self.device),
        }
    
    def insert(
        self,
        low_obs: torch.Tensor,
        high_obs: torch.Tensor,
        action: torch.Tensor,
        belief_h: torch.Tensor,
        belief_h_pred: torch.Tensor,
        belief_h_new: torch.Tensor,
    ):
        """Insert new transitions into the buffer.
        
        Args:
            low_obs: [num_envs, num_steps, low_obs_dim] or [num_envs, low_obs_dim]
            high_obs: [num_envs, num_steps, high_obs_dim] or [num_envs, high_obs_dim]
            action: [num_envs, num_steps, action_dim] or [num_envs, action_dim]
            belief_h: [num_envs, num_steps, belief_dim] or [num_envs, belief_dim]
            belief_h_pred: [num_envs, num_steps, belief_dim] or [num_envs, belief_dim]
            belief_h_new: [num_envs, num_steps, belief_dim] or [num_envs, belief_dim]
        """
        if self.replay_bufs is None:
            num_envs = low_obs.shape[0]
            self._initialize_buffer(num_envs)
        
        # Ensure 3D: [num_envs, num_steps, dim]
        if low_obs.dim() == 2:
            low_obs = low_obs.unsqueeze(1)
        if high_obs.dim() == 2:
            high_obs = high_obs.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)
        if belief_h.dim() == 2:
            belief_h = belief_h.unsqueeze(1)
        if belief_h_pred.dim() == 2:
            belief_h_pred = belief_h_pred.unsqueeze(1)
        if belief_h_new.dim() == 2:
            belief_h_new = belief_h_new.unsqueeze(1)
        
        num_inputs = low_obs.shape[1]
        
        def _insert_into_buffer(r_buf, i_buf):
            """Insert data into buffer, handling circular wrapping."""
            end_idx = self.step + num_inputs
            if end_idx > self.buffer_size:
                r_buf[:, self.step : self.buffer_size] = i_buf[:, : self.buffer_size - self.step]
                r_buf[:, : end_idx - self.buffer_size] = i_buf[:, self.buffer_size - self.step :]
            else:
                r_buf[:, self.step : end_idx] = i_buf
        
        _insert_into_buffer(self.replay_bufs["low_obs"], low_obs)
        _insert_into_buffer(self.replay_bufs["high_obs"], high_obs)
        _insert_into_buffer(self.replay_bufs["action"], action)
        _insert_into_buffer(self.replay_bufs["belief_h"], belief_h)
        _insert_into_buffer(self.replay_bufs["belief_h_pred"], belief_h_pred)
        _insert_into_buffer(self.replay_bufs["belief_h_new"], belief_h_new)
        
        self.num_transitions = min(self.buffer_size, self.num_transitions + num_inputs)
        self.step = (self.step + num_inputs) % self.buffer_size
    
    def mini_batch_generator(
        self,
        sequence_length: int,
        num_mini_batches: int,
        mini_batch_size: int,
    ):
        """Yield mini-batches of sequences for training.
        
        Args:
            sequence_length: Length of sequences to sample
            num_mini_batches: Number of mini-batches to generate
            mini_batch_size: Size of each mini-batch
        
        Yields:
            Tuple of (low_obs, high_obs, action, belief_h, belief_h_pred, belief_h_new)
            Each has shape [mini_batch_size, sequence_length, dim]
        """
        assert self.replay_bufs is not None, "Replay buffer is not initialized."
        
        # Simple random sampling (can be improved with reset-aware sampling)
        for _ in range(num_mini_batches):
            # Sample random environments and start indices
            env_indices = torch.randint(0, self.num_envs, (mini_batch_size,), device=self.device)
            start_indices = torch.randint(
                0, max(1, self.num_transitions - sequence_length + 1),
                (mini_batch_size,), device=self.device
            )
            
            # Extract sequences
            low_obs_batch = []
            high_obs_batch = []
            action_batch = []
            belief_h_batch = []
            belief_h_pred_batch = []
            belief_h_new_batch = []
            
            for i, (env_idx, start_idx) in enumerate(zip(env_indices, start_indices)):
                end_idx = start_idx + sequence_length
                low_obs_batch.append(self.replay_bufs["low_obs"][env_idx, start_idx:end_idx])
                high_obs_batch.append(self.replay_bufs["high_obs"][env_idx, start_idx:end_idx])
                action_batch.append(self.replay_bufs["action"][env_idx, start_idx:end_idx])
                belief_h_batch.append(self.replay_bufs["belief_h"][env_idx, start_idx:end_idx])
                belief_h_pred_batch.append(self.replay_bufs["belief_h_pred"][env_idx, start_idx:end_idx])
                belief_h_new_batch.append(self.replay_bufs["belief_h_new"][env_idx, start_idx:end_idx])
            
            yield (
                torch.stack(low_obs_batch, dim=0),
                torch.stack(high_obs_batch, dim=0),
                torch.stack(action_batch, dim=0),
                torch.stack(belief_h_batch, dim=0),
                torch.stack(belief_h_pred_batch, dim=0),
                torch.stack(belief_h_new_batch, dim=0),
            )


@configclass
class FlowReplayBufferCfg:
    class_type: type[FlowReplayBuffer] = FlowReplayBuffer
    buffer_size: int = 100_000
    
    def construct_from_cfg(
        self,
        low_obs_dim: int,
        high_obs_dim: int,
        action_dim: int,
        belief_dim: int,
        device: str,
        num_envs: int,
    ) -> FlowReplayBuffer:
        return self.class_type(
            low_obs_dim=low_obs_dim,
            high_obs_dim=high_obs_dim,
            action_dim=action_dim,
            belief_dim=belief_dim,
            buffer_size=self.buffer_size,
            device=device,
        )
