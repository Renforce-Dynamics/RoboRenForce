from __future__ import annotations

import os
import time
import torch
from collections import deque
from RoboRenForce import configclass
from dataclasses import MISSING

from RoboRenForce.utils.logging import timeit
from RoboRenForce.algorithms.imitation.distillation import Distillation, DistillationCfg
from RoboRenForce.components.actor.student_teacher import StudentTeacher, StudentTeacherCfg
from RoboRenForce.components.normalizer import NormalizerBaseCfg
from RoboRenForce.runners.logger import LoggerBaseCfg
from RoboRenForce.runners.base_runner import BaseRunner, BaseRunnerCfg
from RoboRenForce.prototype.gym import RoboRenForceVecEnv


class DistillationRunner(BaseRunner):
    """
    On-policy runner for training and evaluation of teacher-student distillation.
    
    The student network uses policy observations while the teacher network uses
    teacher/privileged observations. The student is trained via behavior cloning
    to match the teacher's actions.
    """
    
    env: "RoboRenForceVecEnv"
    cfg: "DistillationRunnerCfg"
    alg: Distillation
    policy: StudentTeacher
    
    def __init__(
        self,
        train_cfg: "DistillationRunnerCfg",
        env: "RoboRenForceVecEnv",
        log_dir=None,
        device="cpu",
    ):
        self.alg_cfg: DistillationCfg = train_cfg.algorithm
        self.policy_cfg: StudentTeacherCfg = train_cfg.policy
        super().__init__(train_cfg=train_cfg, env=env, log_dir=log_dir, device=device)
    
    def init_components(self):
        """Initialize all components for distillation training."""
        # Get observation dimensions
        student_obs_dim = self.env.dim_params["policy_dim"]
        teacher_obs_dim = self.env.dim_params.get("teacher_dim", student_obs_dim)
        action_dim = self.env.num_actions
        
        # Construct student-teacher policy
        self.policy = self.policy_cfg.construct_from_cfg(
            state_dim=student_obs_dim,
            action_dim=action_dim,
            teacher_state_dim=teacher_obs_dim,
        )
        self.policy.to(self.device)
        
        # Check if teacher is loaded
        if not self.policy.loaded_teacher:
            raise ValueError(
                "Teacher model parameters not loaded. "
                "Please load a teacher model to distill using runner.load() or "
                "policy.load_state_dict() before calling learn()."
            )
        
        # Construct algorithm
        self.alg = self.alg_cfg.construct_from_cfg(
            policy=self.policy,
            device=self.device
        )
        
        # Initialize storage
        self.alg.init_storage(
            self.env.num_envs,
            self.cfg.num_steps_per_env,
            [student_obs_dim],
            [teacher_obs_dim],
            [action_dim],
        )
        
        # Normalizers (optional, for compatibility)
        self.obs_normalizer = self.cfg.obs_normalize_cfg.construct_from_cfg(
            shape=student_obs_dim
        )
        self.obs_normalizer.to(self.device)
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Start learning process.
        
        Args:
            num_learning_iterations: Number of learning iterations
            init_at_random_ep_len: If True, randomize initial episode lengths
        """
        self.logger.init_logger()
        
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )
        
        # Book keeping
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        
        self._learn(num_learning_iterations=num_learning_iterations)
    
    def _learn(self, num_learning_iterations: int):
        """Internal learning loop.
        
        Args:
            num_learning_iterations: Number of learning iterations
        """
        ep_infos = []
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        # Get initial observations
        obs, extras = self.env.get_observations()
        teacher_obs = extras["observations"].get("teacher", obs)
        obs, teacher_obs = obs.to(self.device), teacher_obs.to(self.device)
        
        # Normalize observations
        obs = self.obs_normalizer(obs)
        
        self.train_mode()  # Switch to train mode
        
        for it in range(start_iter, tot_iter):
            # Sample rollout
            sample_infos, obs, teacher_obs = self.sample_rollout(
                ep_infos, obs, teacher_obs
            )
            collection_time = sample_infos["collection_time"]
            
            # Update policy
            start = time.time()
            alg_update_infos = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            
            self.current_learning_iteration = it
            
            # Logging
            if self.logger.log_dir is not None:
                log_locals = locals()
                self.logger.log(self, log_locals)
            
            # Save model
            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            
            ep_infos.clear()
        
        # Save final model
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
    
    @timeit("collection_time")
    def sample_rollout(self, ep_infos, obs, teacher_obs, **kwargs):
        """Sample a rollout from the environment.
        
        Args:
            ep_infos: Episode information list
            obs: Current student observations [num_envs, obs_dim]
            teacher_obs: Current teacher observations [num_envs, teacher_obs_dim]
            
        Returns:
            Tuple of (sample_infos, next_obs, next_teacher_obs)
        """
        rollout_datas = []
        
        with torch.inference_mode():
            for i in range(self.cfg.num_steps_per_env):
                # Get actions from student policy (using teacher obs for target)
                actions = self.alg.act(obs, critic_obs=teacher_obs)
                
                # Step environment
                obs_next, reward, done, infos = self.env.step(
                    actions.to(self.env.device)
                )
                
                # Get next observations
                teacher_obs_next = infos["observations"].get("teacher", obs_next)
                obs_next, teacher_obs_next = (
                    obs_next.to(self.device),
                    teacher_obs_next.to(self.device),
                )
                
                # Normalize observations
                obs_next = self.obs_normalizer(obs_next)
                
                # Process environment step
                self.process_env_step(reward, done, infos)
                
                # Store rollout data
                rollout_datas.append((obs, teacher_obs, actions, reward, done, infos))
                
                # Book keeping
                if self.logger.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])
                    
                    self.cur_reward_sum += reward
                    self.cur_episode_length += 1
                    
                    new_ids = (done > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(
                        self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self.lenbuffer.extend(
                        self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0
                
                # Update for next iteration
                obs = obs_next
                teacher_obs = teacher_obs_next
        
        # Process rollout
        process_infos = self.process_rollout(rollout_datas)
        sample_infos = {
            "cur_reward_sum": self.cur_reward_sum,
            "cur_episode_length": self.cur_episode_length,
        }
        sample_infos.update(process_infos)
        
        return sample_infos, obs, teacher_obs
    
    def process_env_step(self, reward, done, infos, **kwargs):
        """Process environment step.
        
        Args:
            reward: Rewards [num_envs]
            done: Done flags [num_envs]
            infos: Additional information dict
        """
        self.alg.process_env_step(reward, done, infos)
    
    def save(self, path, infos=None):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            infos: Additional information to save
        """
        saved_dict = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        self.logger.save_model(saved_dict, path, self.current_learning_iteration)
    
    def load(self, path, load_optimizer=True):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Additional information from checkpoint
        """
        loaded_dict = torch.load(path, map_location=self.device)
        
        # Load policy state dict (handles both RL and distillation checkpoints)
        resume_training = self.policy.load_state_dict(
            loaded_dict["model_state_dict"],
            strict=False
        )
        
        # Update loaded_teacher flag
        self.policy.loaded_teacher = True
        
        # Load optimizer if requested and available
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        
        # Update iteration if available
        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        
        return loaded_dict.get("infos", None)
    
    def get_inference_policy(self, device=None):
        """Get inference policy for evaluation.
        
        Args:
            device: Device to move policy to
            
        Returns:
            Inference policy function
        """
        if device is not None:
            self.policy.to(device)
        return lambda x: self.policy.act_inference(self.obs_normalizer(x))
    
    def train_mode(self):
        """Set to training mode."""
        self.policy.train()
        self.obs_normalizer.train()
    
    def eval_mode(self):
        """Set to evaluation mode."""
        self.policy.eval()
        self.obs_normalizer.eval()


@configclass
class DistillationRunnerCfg(BaseRunnerCfg):
    class_type: type[DistillationRunner] = DistillationRunner
    
    seed: int = 42
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    resume: bool = False
    load_checkpoint: str = "model_.*.pt"
    
    # Policy configuration (StudentTeacher)
    policy: StudentTeacherCfg = MISSING
    
    # Algorithm configuration
    algorithm: DistillationCfg = MISSING
    
    # Normalizer configuration
    obs_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()
    
    # Logger configuration
    logger_cfg: LoggerBaseCfg = LoggerBaseCfg()
