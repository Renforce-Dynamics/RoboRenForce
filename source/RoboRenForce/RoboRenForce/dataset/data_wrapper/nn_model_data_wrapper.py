"""World model data wrapper implementation."""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Tuple
import random
from RoboRenForce import configclass
from RoboRenForce.dataset.data_wrapper.offline_data_wrapper_base import (
    OfflineDataWrapperBase,
    OfflineDataWrapperBaseCfg,
)
from dataclasses import MISSING


class WorldModelDataWrapper(OfflineDataWrapperBase):
    """World model data wrapper.
    
    Used for nn model training. Provides batches of (s, a, r, s', done) sequences.
    """
    
    cfg: "WorldModelDataWrapperCfg"
    
    def __init__(
        self,
        cfg: "WorldModelDataWrapperCfg",
        data_loader,
        device: str = "cpu",
    ):
        super().__init__(cfg, data_loader, device)
        self.sequence_length = cfg.sequence_length
    
    def get_batch(
        self,
        batch_size: int,
        sampling_strategy: str = "random",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Get a batch of data for nn model training.
        
        Args:
            batch_size: Batch size
            sampling_strategy: Sampling strategy ("random", "trajectory")
            sequence_length: Sequence length (overrides cfg.sequence_length if provided)
        
        Returns:
            Dict[str, torch.Tensor]: Batch data with keys:
                - "dynamic": (batch_size, sequence_length, dynamic_dim) - State sequence
                - "action": (batch_size, sequence_length, action_dim) - Action sequence
                - "next_dynamic": (batch_size, sequence_length, dynamic_dim) - Next state
                - "reward": (batch_size, sequence_length, 1) - Reward
                - "termination": (batch_size, sequence_length, 1) - Termination flag
        """
        sequence_length = kwargs.get("sequence_length", self.sequence_length)
        
        batch_dynamic = []
        batch_action = []
        batch_next_dynamic = []
        batch_reward = []
        batch_termination = []
        
        for _ in range(batch_size):
            # Sample a trajectory
            traj_idx = random.randint(0, len(self._trajectories) - 1)
            traj = self._trajectories[traj_idx]
            
            # Get trajectory length
            traj_length = len(traj[list(traj.keys())[0]])
            
            # Sample start index
            if traj_length <= sequence_length:
                start_idx = 0
                actual_seq_len = traj_length
            else:
                start_idx = random.randint(0, traj_length - sequence_length)
                actual_seq_len = sequence_length
            
            # Extract sequence
            dynamic_seq = traj.get("dynamic", traj.get("policy", None))
            if dynamic_seq is None:
                raise ValueError("'dynamic' or 'policy' field is required for nn model training")
            
            action_seq = traj.get("action")
            if action_seq is None:
                raise ValueError("'action' field is required for nn model training")
            
            # Get sequences
            dynamic = dynamic_seq[start_idx:start_idx + actual_seq_len]
            action = action_seq[start_idx:start_idx + actual_seq_len]
            
            # Get next states (shift by 1)
            if start_idx + actual_seq_len < traj_length:
                next_dynamic = dynamic_seq[start_idx + 1:start_idx + actual_seq_len + 1]
            else:
                # Last state: next state is the same (terminal)
                next_dynamic = np.concatenate([
                    dynamic_seq[start_idx + 1:],
                    dynamic_seq[-1:],  # Repeat last state
                ])
            
            # Get rewards
            if "rewards" in traj:
                reward = traj["rewards"][start_idx:start_idx + actual_seq_len]
                if reward.ndim == 1:
                    reward = reward[:, np.newaxis]
            elif "reward" in traj:
                reward = traj["reward"][start_idx:start_idx + actual_seq_len]
                if reward.ndim == 1:
                    reward = reward[:, np.newaxis]
            else:
                reward = np.zeros((actual_seq_len, 1))
            
            # Get termination flags
            if "termination" in traj:
                termination = traj["termination"][start_idx:start_idx + actual_seq_len]
                if termination.ndim == 1:
                    termination = termination[:, np.newaxis]
            else:
                termination = np.zeros((actual_seq_len, 1), dtype=bool)
            
            # Pad if necessary
            if actual_seq_len < sequence_length:
                pad_len = sequence_length - actual_seq_len
                dynamic = np.pad(dynamic, ((0, pad_len), (0, 0)), mode="edge")
                action = np.pad(action, ((0, pad_len), (0, 0)), mode="edge")
                next_dynamic = np.pad(next_dynamic, ((0, pad_len), (0, 0)), mode="edge")
                reward = np.pad(reward, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
                termination = np.pad(termination, ((0, pad_len), (0, 0)), mode="constant", constant_values=True)
            
            batch_dynamic.append(dynamic)
            batch_action.append(action)
            batch_next_dynamic.append(next_dynamic)
            batch_reward.append(reward)
            batch_termination.append(termination)
        
        # Stack into tensors
        batch = {
            "dynamic": torch.from_numpy(np.stack(batch_dynamic)).to(self.device),
            "action": torch.from_numpy(np.stack(batch_action)).to(self.device),
            "next_dynamic": torch.from_numpy(np.stack(batch_next_dynamic)).to(self.device),
            "reward": torch.from_numpy(np.stack(batch_reward)).to(self.device),
            "termination": torch.from_numpy(np.stack(batch_termination)).to(self.device),
        }
        
        return batch
    
    def get_trajectory(self, traj_idx: int) -> Dict[str, torch.Tensor]:
        """Get a specific trajectory.
        
        Args:
            traj_idx: Trajectory index
        
        Returns:
            Dict[str, torch.Tensor]: Trajectory data with shape (T, ...)
        """
        if traj_idx >= len(self._trajectories):
            raise IndexError(f"Trajectory index {traj_idx} out of range")
        
        traj = self._trajectories[traj_idx]
        traj_tensors = {}
        for key, value in traj.items():
            traj_tensors[key] = torch.from_numpy(value).to(self.device)
        
        return traj_tensors


@configclass
class WorldModelDataWrapperCfg(OfflineDataWrapperBaseCfg):
    """Configuration for nn model data wrapper."""
    
    class_type: type[WorldModelDataWrapper] = WorldModelDataWrapper
    
    sequence_length: int = 1
    """Sequence length for nn model training."""
