from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import Dict
from RoboRenForce import configclass
from RoboRenForce.utils.logging import timeit
from dataclasses import MISSING

from RoboRenForce.buffer.replay_buffer_base import \
    ReplayBufferBase, ReplayBufferBaseCfg
from RoboRenForce.components.nn_models import system_dynamics
from RoboRenForce.runners.offline import \
    OfflineRunnerBase, OfflineRunnerBaseCfg
from RoboRenForce.components.nn_models.system_dynamics import \
    SystemDynamicsBase, SystemDynamicsBaseCfg
from RoboRenForce.algorithms.nn_model_trainer import \
    SystemDynamicsTrainer, SystemDynamicsTrainerCfg

class OfflineSystemDynamicsRunner(OfflineRunnerBase):
    """Offline nn model training runner.
    
    Trains system dynamics models from offline data without environment interaction.
    Similar to MBPOOnPolicyRunner but uses offline data instead of online rollouts.
    """
    
    cfg                 : "OfflineSystemDynamicsRunnerCfg"
    system_dynamics     : "SystemDynamicsBase"
    trainer             : "SystemDynamicsTrainer"
    replay_buffer       : "ReplayBufferBase"
    
    def __init__(
        self,
        train_cfg: "OfflineSystemDynamicsRunnerCfg",
        log_dir=None,
        device: str = "cpu",
    ):
        super().__init__(train_cfg, log_dir, device)
        self.system_dynamics_cfg = train_cfg.system_dynamics_cfg
    
    def init_components(self):
        """Initialize components for nn model training."""
        super().init_components()
        dim_params = self.data_cli.dim_params
        self.system_dynamics = self.system_dynamics_cfg.construct_from_cfg(
            dim_params=dim_params,
            device=self.device
        )
        self.system_dynamics.to(self.device)
        
        self.replay_buffer = self.cfg.replay_buffer_cfg.construct_from_cfg(
            dim_params=dim_params, device=self.device, num_envs=1
        )
        self.trainer = self.cfg.trainer_cfg.construct_from_cfg(
            replay_buffer = self.replay_buffer,
            system_dynamics = self.system_dynamics
        )
    
    @timeit("collection_time")
    def gather(self) -> Dict[str, float]:
        """Gather data from data_cli to replay buffer.
        
        This method samples trajectories from the offline dataset and inserts them
        into the replay buffer for training the system dynamics model.
        
        Returns:
            Dict[str, float]: Statistics about the data gathering process, including:
                - "num_trajectories": Number of trajectories gathered
                - "total_steps": Total number of steps gathered
        """
        num_trajectories = len(self.data_cli)
        
        if self.cfg.num_steps_per_env is not None:
            num_trajectories = min(num_trajectories, self.cfg.num_steps_per_env)
        
        total_steps = 0
        for traj_idx in range(num_trajectories):
            # Get trajectory from data wrapper
            traj = self.data_cli.get_trajectory(traj_idx)
            
            # Extract required fields
            dynamic = traj.get("dynamic", traj.get("policy", None))
            if dynamic is None: continue  # Skip trajectories without dynamic/policy field
            
            action = traj.get("action", None)
            if action is None: continue  # Skip trajectories without action field
            
            # Get optional fields
            extension = traj.get("extension", None)
            contact = traj.get("contact", None)
            termination = traj.get("termination", None)
            reward = traj.get("reward", traj.get("rewards", None))
            
            dynamic = dynamic.to(self.device)
            action = action.to(self.device)
            if extension is not None: extension = extension.to(self.device)
            if contact is not None: contact = contact.to(self.device)
            if termination is not None: termination = termination.to(self.device)
            if reward is not None: reward = reward.to(self.device)

            if dynamic.dim() == 2: dynamic = dynamic.unsqueeze(0)  # [1, T, dynamic_dim]
            if action.dim() == 2: action = action.unsqueeze(0)  # [1, T, action_dim]
            if extension is not None and extension.dim() == 2:
                extension = extension.unsqueeze(0)
            if contact is not None and contact.dim() == 2:
                contact = contact.unsqueeze(0)
            if termination is not None and termination.dim() == 2:
                termination = termination.unsqueeze(0)
            if reward is not None and reward.dim() == 2:
                reward = reward.unsqueeze(0)

            self.replay_buffer.insert(
                dynamic=dynamic,
                action=action,
                extension=extension,
                contact=contact,
                termination=termination,
                reward=reward,
            )
            # Count steps
            total_steps += dynamic.shape[1]
        return {
            "num_trajectories": float(num_trajectories),
            "total_steps": float(total_steps),
        }
    
    @timeit("update_time")
    def update(self) -> Dict[str, float]:
        """Update system dynamics model using trainer.
        
        This method calls the trainer to update the system dynamics model
        using data from the replay buffer.
        
        Returns:
            Dict[str, float]: Training metrics including various loss components:
                - "state_loss": Dynamic state prediction loss
                - "sequence_loss": Sequence prediction loss
                - "bound_loss": Bound regularization loss
                - "extension_loss": Extension prediction loss
                - "contact_loss": Contact prediction loss
                - "termination_loss": Termination prediction loss
                - "reward_loss": Reward prediction loss
        """
        update_info = self.trainer.update_system_dynamics()
        return update_info
    
@configclass
class OfflineSystemDynamicsRunnerCfg(OfflineRunnerBaseCfg):
    class_type              : type[OfflineSystemDynamicsRunner] = OfflineSystemDynamicsRunner
    system_dynamics_cfg     : SystemDynamicsBaseCfg = MISSING
    replay_buffer_cfg       : ReplayBufferBaseCfg = MISSING
    trainer_cfg             : SystemDynamicsTrainerCfg = MISSING
    num_steps_per_env       : int = None # here is used for determine how many traj is used for add to buffer
