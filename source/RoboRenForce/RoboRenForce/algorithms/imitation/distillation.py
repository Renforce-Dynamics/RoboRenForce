from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.components.actor.student_teacher import StudentTeacher
from RoboRenForce.buffer.online_rollout.rollout_storage import RolloutStorage
from RoboRenForce.utils.logging import timeit
from RoboRenForce.algorithms.algorithm_base import AlgorithmBase, AlgorithmBaseCfg


class Distillation(AlgorithmBase):
    """
    Distillation algorithm for training a student model to mimic a teacher model.
    
    The student network uses policy observations while the teacher network uses
    teacher/privileged observations. The student is trained via behavior cloning
    to match the teacher's actions.
    """
    
    policy: StudentTeacher
    
    def __init__(
        self,
        cfg: "DistillationCfg",
        policy: StudentTeacher,
        device: str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # Distillation components
        self.policy = policy.to(device)
        self.storage = None  # Initialized later
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=cfg.learning_rate
        )
        
        # Initialize transition
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None
        
        # Initialize loss function
        loss_fn_dict = {
            "mse": F.mse_loss,
            "huber": F.huber_loss,
        }
        if cfg.loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[cfg.loss_type]
        else:
            raise ValueError(
                f"Unknown loss type: {cfg.loss_type}. "
                f"Supported types are: {list(loss_fn_dict.keys())}"
            )
        
        self.num_updates = 0
    
    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape,
        teacher_obs_shape,
        actions_shape,
    ):
        """Initialize rollout storage.
        
        Args:
            num_envs: Number of parallel environments
            num_transitions_per_env: Number of transitions per environment
            obs_shape: Shape of student observations (list of ints)
            teacher_obs_shape: Shape of teacher observations (list of ints)
            actions_shape: Shape of actions (list of ints)
        """
        # Use privileged_obs_shape to store teacher observations
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            obs_shape,
            teacher_obs_shape,  # Store teacher observations here
            actions_shape,
            self.device,
        )
    
    @torch.no_grad()
    def act(self, obs, critic_obs=None):
        """Sample actions from student policy.
        
        Args:
            obs: Student observations [num_envs, obs_dim]
            critic_obs: Teacher observations [num_envs, teacher_obs_dim] (optional)
            
        Returns:
            Actions [num_envs, action_dim]
        """
        # Compute student actions
        self.transition.actions = self.policy.act(obs).detach()
        
        # Store teacher observations for later use in update
        # We use critic_observations field to store teacher observations
        if critic_obs is not None:
            self.transition.critic_observations = critic_obs
        else:
            # If not provided, use student obs (fallback)
            self.transition.critic_observations = obs
        
        # Record student observations
        self.transition.observations = obs
        
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        """Process environment step and store transition.
        
        Args:
            rewards: Rewards [num_envs]
            dones: Done flags [num_envs]
            infos: Additional information dict containing observations
        """
        # Update normalizers
        if "observations" in infos:
            obs_dict = infos["observations"]
            # Get student observations for normalization update
            student_obs = obs_dict.get("policy", None)
            if student_obs is not None:
                self.policy.update_normalization(student_obs)
            
            # Update teacher observations in transition if available
            teacher_obs = obs_dict.get("teacher", None)
            if teacher_obs is not None:
                self.transition.critic_observations = teacher_obs
        
        # Record rewards and dones
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # Store transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        
        # Reset policy (for recurrent policies)
        self.policy.reset(dones)
    
    @timeit("update_time")
    def update(self):
        """Perform one update step of the distillation algorithm.
        
        Returns:
            Dictionary containing loss information
        """
        self.num_updates += 1
        mean_behavior_loss = 0.0
        loss = 0.0
        cnt = 0
        
        for epoch in range(self.cfg.num_learning_epochs):
            # Reset policy hidden states if recurrent
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            
            # Iterate over rollout data
            for batch in self.storage.mini_batch_generator(
                self.cfg.num_mini_batches,
                self.cfg.num_learning_epochs
            ):
                obs_batch, teacher_obs_batch, actions_batch, _, _, _, _, _, _, _ = batch
                
                # Get student actions (for gradient computation)
                student_actions = self.policy.act_inference(obs_batch)
                
                # Get teacher actions (target) from teacher observations
                teacher_actions = self.policy.evaluate(teacher_obs_batch)
                
                # Behavior cloning loss
                behavior_loss = self.loss_fn(student_actions, teacher_actions)
                
                # Accumulate loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1
                
                # Gradient step (with gradient accumulation)
                if cnt % self.cfg.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    if self.cfg.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.policy.student.parameters(),
                            self.cfg.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0.0
        
        # Average loss
        if cnt > 0:
            mean_behavior_loss /= cnt
        
        # Clear storage
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()
        
        # Construct loss dictionary
        loss_dict = {
            "behavior_loss": mean_behavior_loss,
            "num_updates": self.num_updates,
        }
        
        return loss_dict
    
    def train_mode(self):
        """Set to training mode."""
        self.policy.train()
    
    def test_mode(self):
        """Set to evaluation mode."""
        self.policy.eval()


@configclass
class DistillationCfg(AlgorithmBaseCfg):
    class_type: type[Distillation] = Distillation
    
    # Optimization
    learning_rate: float = 1e-3
    num_learning_epochs: int = 1
    num_mini_batches: int = 1
    gradient_length: int = 15  # Gradient accumulation length
    max_grad_norm: float = None  # None means no clipping
    
    # Loss function
    loss_type: str = "mse"  # "mse" or "huber"
    
    def construct_from_cfg(self, policy: StudentTeacher, device, *args, **kwargs):
        return self.class_type(
            self,
            policy=policy,
            device=device
        )
