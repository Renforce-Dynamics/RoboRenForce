from __future__ import annotations

from typing import Dict, Callable, Optional
import torch
import numpy as np

from .exploration_coefficient import ExplorationCoefficient


def filter_leader(
    tensor: torch.Tensor,
    leader_size: int,
    repeat_idxs: list[int],
    num_blocks: int,
) -> torch.Tensor:
    """
    Filter tensor to keep only leader block (block 0) data.
    
    In Leader-Follower mode, only the leader block's data is kept,
    while follower blocks' data is filtered out.
    
    Args:
        tensor: Input tensor [total_size, ...]
        leader_size: Size of leader block data
        repeat_idxs: List of block indices that were repeated
        num_blocks: Total number of blocks
        
    Returns:
        Filtered tensor containing only leader block data
    """
    # Keep original leader data (first leader_size elements)
    # and filter out repeated blocks
    if len(repeat_idxs) == 1:
        return tensor[:leader_size]
    
    # For each repeated block, we need to filter out its data
    # The structure is: [leader_data, block_1_data, block_2_data, ...]
    keep_indices = list(range(leader_size))
    
    # Skip repeated blocks (they come after leader_size)
    current_size = leader_size
    for r_k in repeat_idxs[1:]:
        # Skip this block's data
        current_size += leader_size
    
    return tensor[:current_size]


class SAPGBatchAugmenter:
    """
    Batch augmenter for SAPG (Split and Aggregate Policy Gradients).
    
    Implements the core aggregation mechanism by:
    1. Selecting blocks to repeat (importance sampling)
    2. Modifying observations with different exploration coefficients
    3. Recomputing values and returns with new exploration coefficients
    4. Concatenating augmented data for policy update
    """
    
    def __init__(
        self,
        expl_coef: ExplorationCoefficient,
        off_policy_ratio: float = 1.0,
        gamma: float = 0.99,
        use_leader_follower: bool = False,
    ):
        """
        Initialize SAPG batch augmenter.
        
        Args:
            expl_coef: Exploration coefficient manager
            off_policy_ratio: Ratio of off-policy blocks to include (controls how many blocks to repeat)
            gamma: Discount factor for return computation
            use_leader_follower: Whether to use Leader-Follower mode (only keep leader block data)
        """
        self.expl_coef = expl_coef
        self.off_policy_ratio = off_policy_ratio
        self.gamma = gamma
        self.use_leader_follower = use_leader_follower
        self.num_blocks = expl_coef.num_blocks
        self.block_size = expl_coef.block_size
    
    def augment_batch(
        self,
        batch_dict: Dict[str, torch.Tensor],
        extras: Dict,
        get_values_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Augment batch data by repeating blocks with different exploration coefficients.
        
        Args:
            batch_dict: Original batch data containing:
                - 'returns': [horizon * num_envs, 1]
                - 'values': [horizon * num_envs, 1]
                - 'obses': [horizon * num_envs, obs_dim]
                - Other fields as needed
            extras: Additional data containing:
                - 'obs': [horizon, num_envs, obs_dim]
                - 'rewards': [horizon, num_envs, 1]
                - 'dones': [horizon, num_envs, 1]
                - 'intr_rewards': [horizon, num_envs, 1] (optional)
                - 'last_obs': [num_envs, obs_dim]
                - 'last_dones': [num_envs, 1]
            get_values_fn: Function to compute values given observations
            
        Returns:
            Augmented batch dictionary with concatenated data from multiple blocks
        """
        # 1. Select blocks to repeat (importance sampling)
        num_repeat = min(self.num_blocks, int(self.off_policy_ratio) + 1)
        repeat_idxs = [0] + self._sample_block_indices(num_repeat - 1)
        
        # 2. Initialize new batch dictionary
        new_batch_dict = {}
        new_returns_list = [batch_dict["returns"]]
        new_values_list = [batch_dict["values"]]
        
        # 3. Process each repeated block (skip block 0, it's the original)
        for r_k in repeat_idxs[1:]:
            # Modify observations with block r_k's exploration coefficient
            mb_obs = self._modify_observations(extras["obs"], r_k)
            
            # Recompute values with modified observations
            mb_values = self._recompute_values(mb_obs, extras, get_values_fn)
            
            # Recompute returns with new exploration coefficient
            mb_returns = self._recompute_returns(extras, mb_values, r_k)
            
            # Flatten and add to lists
            new_returns_list.append(self._flatten_returns(mb_returns))
            new_values_list.append(self._flatten_values(mb_values))
        
        # 4. Concatenate returns and values
        new_batch_dict["returns"] = torch.cat(new_returns_list, dim=0)
        new_batch_dict["values"] = torch.cat(new_values_list, dim=0)
        
        # 5. Augment other fields
        for key, val in batch_dict.items():
            if key not in ["returns", "values"]:
                new_batch_dict[key] = self._augment_tensor(val, repeat_idxs)
        
        # 6. Apply Leader-Follower filtering if enabled
        if self.use_leader_follower:
            original_size = len(batch_dict["returns"])
            new_batch_dict = self._filter_leader(
                new_batch_dict, original_size, repeat_idxs
            )
        
        # 7. Add off-policy mask for tracking
        original_size = len(batch_dict["returns"])
        off_policy_mask = torch.zeros(
            len(new_batch_dict["returns"]), dtype=torch.bool, device=new_batch_dict["returns"].device
        )
        off_policy_mask[original_size:] = True
        new_batch_dict["off_policy_mask"] = off_policy_mask
        
        return new_batch_dict
    
    def _sample_block_indices(self, num_samples: int) -> list[int]:
        """
        Sample block indices for augmentation (excluding block 0).
        
        Args:
            num_samples: Number of blocks to sample
            
        Returns:
            List of block indices
        """
        if num_samples == 0:
            return []
        # Sample from blocks 1 to num_blocks-1 (exclude block 0)
        available_blocks = list(range(1, self.num_blocks))
        if len(available_blocks) == 0:
            return []
        sampled = np.random.choice(available_blocks, size=num_samples, replace=False)
        return [int(x) for x in sampled]
    
    def _modify_observations(
        self, obs: torch.Tensor, block_idx: int
    ) -> torch.Tensor:
        """
        Modify observations by replacing exploration coefficient embeddings.
        
        Args:
            obs: Original observations [horizon, num_envs, obs_dim]
            block_idx: Block index to use for exploration coefficient
            
        Returns:
            Modified observations with new exploration coefficient embeddings
        """
        obs = obs.clone()
        embd_dim = self.expl_coef.embd_dim
        
        # Get embeddings for the specified block
        new_embd = self.expl_coef.get_embeddings_for_block(block_idx)
        
        # Replace the last embd_dim dimensions (exploration coefficient embeddings)
        # Expand new_embd to match obs shape: [horizon, num_envs, embd_dim]
        new_embd_expanded = new_embd.unsqueeze(0).expand(obs.shape[0], -1, -1)
        obs[:, :, -embd_dim:] = new_embd_expanded
        
        return obs
    
    def _recompute_values(
        self,
        obs: torch.Tensor,
        extras: Dict,
        get_values_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Recompute values using modified observations.
        
        Args:
            obs: Modified observations [horizon, num_envs, obs_dim]
            extras: Additional data (for RNN states if needed)
            get_values_fn: Function to compute values
            
        Returns:
            Recomputed values [horizon, num_envs, 1]
        """
        # Flatten observations: [horizon * num_envs, obs_dim]
        flattened_obs = obs.reshape(-1, *obs.shape[2:])
        
        # Compute values in batches to avoid memory issues
        batch_size = 8192
        values_list = []
        for i in range(0, len(flattened_obs), batch_size):
            batch_obs = flattened_obs[i : i + batch_size]
            batch_values = get_values_fn(batch_obs)
            values_list.append(batch_values)
        
        values = torch.cat(values_list, dim=0)
        
        # Reshape back: [horizon, num_envs, 1]
        return values.reshape(*obs.shape[:2], *values.shape[1:])
    
    def _recompute_returns(
        self,
        extras: Dict,
        values: torch.Tensor,
        block_idx: int,
    ) -> torch.Tensor:
        """
        Recompute returns with new exploration coefficient.
        
        Uses GAE (Generalized Advantage Estimation) similar to PPO.
        
        Args:
            extras: Additional data containing rewards, dones, etc.
            values: Recomputed values [horizon, num_envs, 1]
            block_idx: Block index for reward coefficient
            
        Returns:
            Recomputed returns [horizon, num_envs, 1]
        """
        rewards = extras["rewards"]  # [horizon, num_envs, 1]
        dones = extras["dones"]  # [horizon, num_envs, 1]
        horizon = rewards.shape[0]
        num_envs = rewards.shape[1]
        
        # Get reward coefficient for this block
        reward_coef = self.expl_coef.get_reward_coef_for_block(block_idx)
        
        # Compute intrinsic reward term (if available)
        intr_rewards = extras.get("intr_rewards", None)
        if intr_rewards is not None:
            # reward_coef: [num_envs] -> [horizon, num_envs, 1]
            intr_reward_term = (
                reward_coef.unsqueeze(0).unsqueeze(2).expand(horizon, -1, -1)
                * intr_rewards
            )
        else:
            intr_reward_term = 0
        
        # Get last values for bootstrap
        last_values = extras.get("last_values", None)
        if last_values is None:
            # Use last value from values tensor
            last_values = values[-1]  # [num_envs, 1]
        
        # Compute returns using GAE
        # returns[t] = rewards[t] + γ * (1 - done[t]) * returns[t+1]
        returns = torch.zeros_like(rewards)
        next_values = last_values  # [num_envs, 1]
        
        for step in reversed(range(horizon)):
            # Compute return for this step
            next_is_not_terminal = 1.0 - dones[step].float()  # [num_envs, 1]
            
            # Add intrinsic reward if available
            if intr_rewards is not None:
                total_reward = rewards[step] + intr_reward_term[step]
            else:
                total_reward = rewards[step]
            
            returns[step] = total_reward + self.gamma * next_is_not_terminal * next_values
            next_values = values[step]  # Use current value for next iteration
        
        return returns
    
    def _flatten_returns(self, returns: torch.Tensor) -> torch.Tensor:
        """Flatten returns from [horizon, num_envs, 1] to [horizon * num_envs, 1]."""
        return returns.reshape(-1, *returns.shape[2:])
    
    def _flatten_values(self, values: torch.Tensor) -> torch.Tensor:
        """Flatten values from [horizon, num_envs, 1] to [horizon * num_envs, 1]."""
        return values.reshape(-1, *values.shape[2:])
    
    def _augment_tensor(
        self, tensor: torch.Tensor, repeat_idxs: list[int]
    ) -> torch.Tensor:
        """
        Augment tensor by repeating it for each block.
        
        Args:
            tensor: Original tensor [size, ...]
            repeat_idxs: List of block indices to repeat
            
        Returns:
            Augmented tensor [size * len(repeat_idxs), ...]
        """
        return torch.cat([tensor] * len(repeat_idxs), dim=0)
    
    def _filter_leader(
        self,
        batch_dict: Dict[str, torch.Tensor],
        leader_size: int,
        repeat_idxs: list[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Filter batch dictionary to keep only leader block data (Leader-Follower mode).
        
        Args:
            batch_dict: Batch dictionary to filter
            leader_size: Size of leader block data
            repeat_idxs: List of block indices that were repeated
            
        Returns:
            Filtered batch dictionary
        """
        filtered_dict = {}
        for key, val in batch_dict.items():
            if key == "off_policy_mask":
                # Keep mask but adjust size
                filtered_dict[key] = val[:leader_size]
            else:
                filtered_dict[key] = filter_leader(
                    val, leader_size, repeat_idxs, self.num_blocks
                )
        return filtered_dict
