from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import MISSING
from typing import Dict, Optional
from RoboRenForce import configclass
from RoboRenForce.utils.template import ClassTemplateBaseCfg
from RoboRenForce.buffer.online_rollout.belief_rollout_storage import BeliefRolloutStorage
import torch.optim as optim


class FlowModelTrainer:
    """Trainer for belief flow models (f_theta and g_phi).
    
    Trains two models:
    1. belief_flow: f_theta(h, low_obs, action_prev) -> dh
    2. belief_obs_update: g_phi(h_pred, high_obs) -> Δh
    """
    
    cfg: "FlowModelTrainerCfg"
    
    def __init__(
        self,
        cfg: "FlowModelTrainerCfg",
        replay_buffer,
        belief_flow: nn.Module,
        belief_obs_update: nn.Module,
    ) -> None:
        self.cfg = cfg
        self.replay_buffer: BeliefRolloutStorage = replay_buffer
        self.belief_flow = belief_flow
        self.belief_obs_update = belief_obs_update
        
        # Optimizers for both models
        self.flow_optimizer = optim.Adam(
            self.belief_flow.parameters(),
            lr=self.cfg.flow_learning_rate,
            weight_decay=self.cfg.flow_weight_decay,
        )
        self.obs_update_optimizer = optim.Adam(
            self.belief_obs_update.parameters(),
            lr=self.cfg.obs_update_learning_rate,
            weight_decay=self.cfg.obs_update_weight_decay,
        )
    
    def update_flow_models(self):
        """Update both belief flow and observation update models from replay buffer."""
        mean_losses = {
            "flow_loss": 0.0,
            "obs_update_loss": 0.0,
        }
        num_updates = 0
        
        # Use flow_model_batch_generator for sequence-based training
        generator = self.replay_buffer.flow_model_batch_generator(
            sequence_length=self.cfg.sequence_length,
            num_mini_batches=self.cfg.num_mini_batches,
            mini_batch_size=self.cfg.mini_batch_size,
        )
        
        for batch in generator:
            (
                low_obs_batch,
                slow_obs_batch,
                action_batch,
                belief_h_batch,
                belief_h_pred_batch,
                belief_h_new_batch,
            ) = batch
            
            # Train belief_flow: f_theta(h, low_obs, action_prev) -> dh
            # Target: h_pred should match the next h (or h_new if slow_obs was available)
            flow_loss = self._compute_flow_loss(
                belief_h_batch,
                low_obs_batch,
                action_batch,
                belief_h_pred_batch,
            )
            
            # Train belief_obs_update: g_phi(h_pred, slow_obs) -> Δh
            # Target: h_new = h_pred + g_phi(h_pred, slow_obs) should match ground truth
            obs_update_loss = self._compute_obs_update_loss(
                belief_h_pred_batch,
                slow_obs_batch,
                belief_h_new_batch,
            )
            
            # Update belief_flow
            self.flow_optimizer.zero_grad()
            flow_loss.backward()
            nn.utils.clip_grad_norm_(
                self.belief_flow.parameters(), self.cfg.max_grad_norm
            )
            self.flow_optimizer.step()
            
            # Update belief_obs_update
            self.obs_update_optimizer.zero_grad()
            obs_update_loss.backward()
            nn.utils.clip_grad_norm_(
                self.belief_obs_update.parameters(), self.cfg.max_grad_norm
            )
            self.obs_update_optimizer.step()
            
            mean_losses["flow_loss"] += float(flow_loss.detach())
            mean_losses["obs_update_loss"] += float(obs_update_loss.detach())
            num_updates += 1
        
        if num_updates == 0:
            return {k: 0.0 for k in mean_losses}
        
        for k in mean_losses:
            mean_losses[k] /= num_updates
        
        return mean_losses
    
    def _compute_flow_loss(
        self,
        h: torch.Tensor,  # [B, T, belief_dim]
        low_obs: torch.Tensor,  # [B, T, low_obs_dim]
        action: torch.Tensor,  # [B, T, action_dim]
        h_pred_target: torch.Tensor,  # [B, T, belief_dim]
    ) -> torch.Tensor:
        """Compute loss for belief flow model.
        
        Predicts: h_pred = h + f_theta(h, low_obs, action_prev)
        Target: h_pred_target (from rollout)
        """
        B, T, _ = h.shape
        
        # Use previous action (shift by 1)
        action_prev = torch.cat([torch.zeros(B, 1, action.shape[-1], device=action.device), action[:, :-1]], dim=1)
        
        # Predict dh for each timestep
        dh_pred = self.belief_flow(
            h.reshape(B * T, -1),
            low_obs.reshape(B * T, -1),
            action_prev.reshape(B * T, -1),
        )
        dh_pred = dh_pred.reshape(B, T, -1)
        
        # Target: h_pred_target - h
        dh_target = h_pred_target - h
        
        # MSE loss
        flow_loss = nn.functional.mse_loss(dh_pred, dh_target)
        
        return flow_loss
    
    def _compute_obs_update_loss(
        self,
        h_pred: torch.Tensor,  # [B, T, belief_dim]
        slow_obs: torch.Tensor,  # [B, T, slow_obs_dim] or None
        h_new_target: torch.Tensor,  # [B, T, belief_dim]
    ) -> torch.Tensor:
        """Compute loss for observation update model.
        
        Predicts: h_new = h_pred + g_phi(h_pred, slow_obs)
        Target: h_new_target (from rollout)
        """
        B, T, _ = h_pred.shape
        
        # Only compute loss where slow_obs is available (non-zero mask)
        # Assume slow_obs is None or has zeros where not available
        if slow_obs is None:
            # If no slow_obs, loss is zero (no update expected)
            return torch.tensor(0.0, device=h_pred.device)
        
        # Check which timesteps have valid slow_obs (non-zero)
        slow_obs_valid = slow_obs.abs().sum(dim=-1) > 1e-6  # [B, T]
        
        if not slow_obs_valid.any():
            return torch.tensor(0.0, device=h_pred.device)
        
        # Predict delta_h for valid timesteps
        h_pred_flat = h_pred.reshape(B * T, -1)
        slow_obs_flat = slow_obs.reshape(B * T, -1)
        delta_h_pred = self.belief_obs_update(h_pred_flat, slow_obs_flat)
        delta_h_pred = delta_h_pred.reshape(B, T, -1)
        
        # Target: h_new_target - h_pred
        delta_h_target = h_new_target - h_pred
        
        # Masked MSE loss (only for valid slow_obs timesteps)
        error = (delta_h_pred - delta_h_target) ** 2
        error = error.sum(dim=-1)  # [B, T]
        obs_update_loss = (error * slow_obs_valid.float()).sum() / (slow_obs_valid.float().sum() + 1e-8)
        
        return obs_update_loss


@configclass
class FlowModelTrainerCfg(ClassTemplateBaseCfg):
    class_type: type[FlowModelTrainer] = FlowModelTrainer
    
    # Training hyperparameters
    flow_learning_rate: float = 1e-3
    flow_weight_decay: float = 0.0
    obs_update_learning_rate: float = 1e-3
    obs_update_weight_decay: float = 0.0
    
    # Batch generation
    sequence_length: int = 2
    num_mini_batches: int = 4
    mini_batch_size: int = 1024
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    def construct_from_cfg(self, *args, **kwargs) -> "FlowModelTrainer":
        return FlowModelTrainer(self, *args, **kwargs)
