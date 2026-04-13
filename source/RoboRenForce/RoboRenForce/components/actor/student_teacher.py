from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from RoboRenForce import configclass
from RoboRenForce.networks.mlp import MLP, MLPCfg
from RoboRenForce.components.actor.actor_base import ActorBase
from RoboRenForce.components.normalizer.normalizer_base import NormalizerBase, NormalizerBaseCfg
from RoboRenForce.utils.template.module_base import ModuleBaseCfg
from dataclasses import MISSING


class StudentTeacher(ActorBase):
    """
    Student-Teacher module for knowledge distillation.
    
    The student network uses policy observations (limited) while the teacher
    network uses teacher observations (typically privileged/rich observations).
    The student is trained to mimic the teacher's actions via behavior cloning.
    """
    
    def __init__(
        self,
        cfg: "StudentTeacherCfg",
        state_dim: int,
        action_dim: int,
        teacher_state_dim: int = None,
    ):
        """
        Args:
            cfg: Configuration for StudentTeacher
            state_dim: Student observation dimension (policy observations)
            action_dim: Action dimension
            teacher_state_dim: Teacher observation dimension (teacher/privileged observations).
                             If None, uses state_dim.
        """
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        
        self.cfg = cfg
        self.teacher_state_dim = teacher_state_dim if teacher_state_dim is not None else state_dim
        self.loaded_teacher = False  # Flag indicating if teacher has been loaded
        
        # Student network: maps student observations to actions
        self.student = cfg.student_backbone_cfg.class_type(
            in_feature=state_dim,
            out_feature=action_dim,
            cfg=cfg.student_backbone_cfg
        )
        
        # Student observation normalizer
        if cfg.student_obs_normalize_cfg.class_type is not nn.Identity:
            self.student_obs_normalizer = cfg.student_obs_normalize_cfg.construct_from_cfg(
                shape=state_dim
            )
        else:
            self.student_obs_normalizer = nn.Identity()
        
        # Teacher network: maps teacher observations to actions
        self.teacher = cfg.teacher_backbone_cfg.class_type(
            in_feature=self.teacher_state_dim,
            out_feature=action_dim,
            cfg=cfg.teacher_backbone_cfg
        )
        self.teacher.eval()  # Teacher is always in eval mode
        
        # Teacher observation normalizer
        if cfg.teacher_obs_normalize_cfg.class_type is not nn.Identity:
            self.teacher_obs_normalizer = cfg.teacher_obs_normalize_cfg.construct_from_cfg(
                shape=self.teacher_state_dim
            )
        else:
            self.teacher_obs_normalizer = nn.Identity()
        self.teacher_obs_normalizer.eval()
        
        # Action noise for exploration during training
        self.noise_std_type = cfg.noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(cfg.init_noise_std * torch.ones(action_dim))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(cfg.init_noise_std * torch.ones(action_dim)))
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        # Action distribution (populated in update_distribution)
        self.distribution = None
        
        # Disable args validation for speedup
        Normal.set_default_validate_args(False)
    
    def update_distribution(self, obs: torch.Tensor):
        """Update the action distribution based on student network output.
        
        Args:
            obs: Student observations [batch_size, state_dim]
        """
        # Compute mean from student network
        mean = self.student(obs)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        
        # Create distribution
        self.distribution = Normal(mean, std)
    
    def forward(self, state: torch.Tensor):
        """Forward pass: compute action distribution.
        
        Args:
            state: Student observations [batch_size, state_dim]
            
        Returns:
            Action distribution
        """
        state = self.student_obs_normalizer(state)
        self.update_distribution(state)
        return self.distribution
    
    @torch.no_grad()
    def act_inference(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic action for inference/play.
        
        Args:
            state: Student observations [batch_size, state_dim]
            
        Returns:
            Deterministic action [batch_size, action_dim]
        """
        state = self.student_obs_normalizer(state)
        return self.student(state)
    
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample action from distribution (for training with exploration).
        
        Args:
            obs: Student observations [batch_size, state_dim]
            
        Returns:
            Sampled action [batch_size, action_dim]
        """
        obs = self.student_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def evaluate(self, teacher_obs: torch.Tensor) -> torch.Tensor:
        """Evaluate teacher network to get target actions.
        
        Args:
            teacher_obs: Teacher observations [batch_size, teacher_state_dim]
            
        Returns:
            Teacher actions [batch_size, action_dim]
        """
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)
        with torch.no_grad():
            return self.teacher(teacher_obs)
    
    def update_normalization(self, student_obs: torch.Tensor):
        """Update student observation normalizer statistics.
        
        Args:
            student_obs: Student observations [batch_size, state_dim]
        """
        if isinstance(self.student_obs_normalizer, NormalizerBase):
            if hasattr(self.student_obs_normalizer, 'update'):
                self.student_obs_normalizer.update(student_obs)
    
    def reset(self, dones=None, hidden_states=None):
        """Reset internal state (for compatibility with recurrent policies).
        
        Args:
            dones: Done flags [batch_size]
            hidden_states: Hidden states (not used for non-recurrent)
        """
        pass
    
    def get_hidden_states(self):
        """Get hidden states (for compatibility with recurrent policies).
        
        Returns:
            None (non-recurrent)
        """
        return None
    
    def detach_hidden_states(self, dones=None):
        """Detach hidden states (for compatibility with recurrent policies).
        
        Args:
            dones: Done flags [batch_size]
        """
        pass
    
    def train(self, mode: bool = True):
        """Set training mode.
        
        Args:
            mode: If True, set to training mode; if False, set to eval mode
        """
        super().train(mode)
        # Teacher is always in eval mode
        self.teacher.eval()
        if hasattr(self.teacher_obs_normalizer, 'eval'):
            self.teacher_obs_normalizer.eval()
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.
        
        This method handles loading from both:
        1. RL training checkpoints (actor parameters → teacher)
        2. Distillation training checkpoints (student + teacher)
        
        Args:
            state_dict: State dictionary of the model
            strict: Whether to strictly enforce key matching
            
        Returns:
            bool: Whether this training resumes a previous training
        """
        # Check if loading from RL training (actor → teacher)
        if any("actor" in key for key in state_dict.keys()):
            # Rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            if isinstance(self.teacher_obs_normalizer, NormalizerBase):
                if hasattr(self.teacher_obs_normalizer, 'load_state_dict'):
                    self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            
            # Set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            if hasattr(self.teacher_obs_normalizer, 'eval'):
                self.teacher_obs_normalizer.eval()
            return False  # Training does not resume
        
        # Check if loading from distillation training (student + teacher)
        elif any("student" in key for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            # Set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            if hasattr(self.teacher_obs_normalizer, 'eval'):
                self.teacher_obs_normalizer.eval()
            return True  # Training resumes
        
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
    
    @property
    def action_mean(self):
        """Get action mean from distribution."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        return self.distribution.mean
    
    @property
    def action_std(self):
        """Get action std from distribution."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """Get entropy from distribution."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        return self.distribution.entropy().sum(dim=-1)


@configclass
class StudentTeacherCfg(ModuleBaseCfg):
    class_type: type[StudentTeacher] = StudentTeacher
    
    # Student network configuration
    student_backbone_cfg: MLPCfg = MISSING
    student_obs_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()
    
    # Teacher network configuration
    teacher_backbone_cfg: MLPCfg = MISSING
    teacher_obs_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()
    
    # Action noise configuration
    noise_std_type: str = "scalar"  # "scalar" or "log"
    init_noise_std: float = 0.1



