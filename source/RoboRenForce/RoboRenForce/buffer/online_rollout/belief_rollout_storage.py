from __future__ import annotations

import torch
from RoboRenForce.buffer.online_rollout.rollout_storage import RolloutStorage


class BeliefRolloutStorage(RolloutStorage):
    """Rollout storage with belief states for flow model training.
    
    Extends RolloutStorage to include:
    - belief states in transitions
    - generator that returns two-frame belief transitions (current and next)
    """
    
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.belief = None  # Current belief state
            self.slow_obs = None  # Slow observation
        
        def clear(self):
            super().clear()
            self.belief = None
            self.slow_obs = None
    
    def __init__(
        self, 
        num_envs, 
        num_transitions_per_env, 
        obs_shape, 
        privileged_obs_shape, 
        actions_shape, 
        belief_dim: int,
        slow_obs_shape=None,
        device="cpu"
    ):
        super().__init__(
            num_envs, 
            num_transitions_per_env, 
            obs_shape, 
            privileged_obs_shape, 
            actions_shape, 
            device
        )
        self.belief_dim = belief_dim
        self.slow_obs_shape = slow_obs_shape
        
        # Belief states: [T, N, belief_dim]
        self.beliefs = torch.zeros(
            num_transitions_per_env, num_envs, belief_dim, device=self.device
        )
        # Store next belief for flow model training (h_pred, h_new)
        self.beliefs_pred = torch.zeros(
            num_transitions_per_env, num_envs, belief_dim, device=self.device
        )
        self.beliefs_new = torch.zeros(
            num_transitions_per_env, num_envs, belief_dim, device=self.device
        )
        
        # Slow observations: [T, N, *slow_obs_shape]
        if slow_obs_shape is not None:
            self.slow_observations = torch.zeros(
                num_transitions_per_env, num_envs, *slow_obs_shape, device=self.device
            )
        else:
            self.slow_observations = None
    
    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        
        # Save current step before parent increments it
        current_step = self.step
        
        # Call parent to add standard fields (this increments self.step)
        super().add_transitions(transition)
        
        # Add belief states at the step that was just added
        if transition.belief is not None:
            self.beliefs[current_step].copy_(transition.belief)
        
        # Add slow observations at the step that was just added
        if self.slow_observations is not None and transition.slow_obs is not None:
            self.slow_observations[current_step].copy_(transition.slow_obs)
    
    def set_belief_pred_new(self, step: int, belief_pred: torch.Tensor, belief_new: torch.Tensor):
        """Set predicted and updated belief states for a given step.
        
        Args:
            step: Step index in the rollout
            belief_pred: Predicted belief state h_pred [N, belief_dim]
            belief_new: Updated belief state h_new [N, belief_dim]
        """
        self.beliefs_pred[step].copy_(belief_pred)
        self.beliefs_new[step].copy_(belief_new)
    
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        
        slow_observations = None
        if self.slow_observations is not None:
            slow_observations = self.slow_observations.flatten(0, 1)

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                slow_batch = slow_observations[batch_idx] if slow_observations is not None else None
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield (obs_batch, slow_batch), critic_observations_batch, \
                    actions_batch, target_values_batch, \
                        advantages_batch, returns_batch, \
                            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch
    
    def flow_model_batch_generator(
        self,
        sequence_length: int,
        num_mini_batches: int,
        mini_batch_size: int,
    ):
        """Generator for flow model training that returns belief sequences.
        
        Returns batches of shape [B, T, ...] where T = ``sequence_length``.
        Each batch contains:
        - low_obs: [B, T, *low_obs_dim]
        - slow_obs: [B, T, *slow_obs_dim] or None
        - action: [B, T, action_dim]
        - belief_h: [B, T, belief_dim] (current belief)
        - belief_h_pred: [B, T, belief_dim] (predicted belief)
        - belief_h_new: [B, T, belief_dim] (updated belief)
        
        Sequences are sampled per-environment; we never cross environment
        boundaries when constructing a sequence.
        """
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValueError(f"sequence_length must be a positive int, got {sequence_length!r}")
        
        T = self.num_transitions_per_env
        N = self.num_envs
        if sequence_length > T:
            raise ValueError(
                f"sequence_length ({sequence_length}) cannot be larger than "
                f"num_transitions_per_env ({T})."
            )
        
        # [T, N, ...]
        observations = self.observations
        actions = self.actions
        beliefs = self.beliefs
        beliefs_pred = self.beliefs_pred
        beliefs_new = self.beliefs_new
        slow_obs = self.slow_observations
        
        # Number of valid starting timesteps per env
        num_starts_per_env = T - sequence_length + 1
        if num_starts_per_env <= 0:
            return
        
        # Precompute all (env, start_t) pairs.
        all_env_indices = torch.arange(N, device=self.device).repeat_interleave(
            num_starts_per_env
        )
        all_start_ts = torch.arange(num_starts_per_env, device=self.device).repeat(N)
        total_sequences = all_env_indices.numel()
        
        for _ in range(num_mini_batches):
            if total_sequences <= mini_batch_size:
                choice = torch.randint(
                    0, total_sequences, (mini_batch_size,), device=self.device
                )
            else:
                choice = torch.randperm(total_sequences, device=self.device)[
                    :mini_batch_size
                ]
            
            batch_envs = all_env_indices[choice]  # [B]
            batch_starts = all_start_ts[choice]   # [B]
            B = batch_envs.shape[0]
            
            # Allocate batches
            low_obs_batch = torch.zeros(
                B, sequence_length, observations.shape[-1], device=self.device
            )
            slow_obs_batch = None
            if slow_obs is not None:
                slow_obs_batch = torch.zeros(
                    B, sequence_length, slow_obs.shape[-1], device=self.device
                )
            action_batch = torch.zeros(
                B, sequence_length, actions.shape[-1], device=self.device
            )
            belief_h_batch = torch.zeros(
                B, sequence_length, self.belief_dim, device=self.device
            )
            belief_h_pred_batch = torch.zeros_like(belief_h_batch)
            belief_h_new_batch = torch.zeros_like(belief_h_batch)
            
            # Fill sequences
            for i, (env_idx, start_t) in enumerate(zip(batch_envs, batch_starts)):
                end_t = start_t + sequence_length
                
                low_obs_batch[i] = observations[start_t:end_t, env_idx]
                if slow_obs_batch is not None:
                    slow_obs_batch[i] = slow_obs[start_t:end_t, env_idx]
                action_batch[i] = actions[start_t:end_t, env_idx]
                belief_h_batch[i] = beliefs[start_t:end_t, env_idx]
                belief_h_pred_batch[i] = beliefs_pred[start_t:end_t, env_idx]
                belief_h_new_batch[i] = beliefs_new[start_t:end_t, env_idx]
            
            yield (
                low_obs_batch,
                slow_obs_batch,
                action_batch,
                belief_h_batch,
                belief_h_pred_batch,
                belief_h_new_batch,
            )
