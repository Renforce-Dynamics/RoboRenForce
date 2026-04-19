from __future__ import annotations

import os
import time
import torch
from collections import deque
from RoboRenForce import configclass
from dataclasses import MISSING

from RoboRenForce.utils.logging import timeit
from RoboRenForce.algorithms import AlgorithmBaseCfg
from RoboRenForce.components.actor_critic_pack import ActorCriticPackCfg
from RoboRenForce.runners.logger import LoggerBaseCfg
from RoboRenForce.runners.on_policy.on_policy_runner import (
    OnPolicyRunner,
    OnPolicyRunnerCfg,
)
from RoboRenForce.prototype.classic import RoboRenForceVecEnv
from RoboRenForce.algorithms.on_policy.sapg.exploration_coefficient import ExplorationCoefficientCfg
from RoboRenForce.algorithms.on_policy.sapg.sapg_augmentation import SAPGBatchAugmenter
from RoboRenForce.algorithms.on_policy.sapg.sapg_ppo import SAPGPPO
from RoboRenForce.components.normalizer import NormalizerBaseCfg


class SAPGOnPolicyRunner(OnPolicyRunner):
    """
    On-policy runner with SAPG (Split and Aggregate Policy Gradients) support.
    
    Extends OnPolicyRunner with:
    - Exploration coefficient management for different blocks
    - Batch augmentation for aggregating experiences from different blocks
    - Support for Leader-Follower mode
    """
    
    def __init__(
        self,
        cfg: "SAPGOnPolicyRunnerCfg",
        env: "RoboRenForceVecEnv",
        log_dir=None,
        device="cpu",
    ):
        self.expl_coef_cfg = cfg.exploration_coef_cfg
        self.off_policy_ratio = cfg.off_policy_ratio
        self.use_leader_follower = cfg.use_leader_follower
        self.use_batch_augmentation = cfg.use_batch_augmentation
        
        super().__init__(train_cfg=cfg, env=env, log_dir=log_dir, device=device)
    
    def init_components(self):
        """Initialize components including exploration coefficient manager."""
        # Store original observation dimensions
        original_policy_dim = self.env.dim_params["policy_dim"]
        original_critic_dim = self.env.dim_params["critic_dim"]
        
        # Initialize exploration coefficient manager
        self.expl_coef = self.expl_coef_cfg.construct_from_cfg(
            num_envs=self.env.num_envs,
            device=self.device,
        )
        
        # Extend observation space to include exploration coefficient embeddings
        embd_dim = self.expl_coef.embd_dim

        self.dim_params = dict(self.env.dim_params)
        self.dim_params["policy_dim"] = original_policy_dim + embd_dim
        self.dim_params["critic_dim"] = original_critic_dim + embd_dim
        
        self.actor_critic = self.policy_cfg.construct_from_cfg(
            dim_params=self.dim_params
        )
        
        # Initialize algorithm (should typically be `SAPGPPO` in SAPG setups)
        self.alg = self.alg_cfg.construct_from_cfg(
            actor_critic=self.actor_critic, device=self.device
        )
        # If algorithm is SAPGPPO, enable SAPG-specific storage; otherwise fall back
        # to standard PPO storage (still functional but without SAPG augmentation).
        if isinstance(self.alg, SAPGPPO):
            self.alg.init_storage(
                self.env.num_envs,
                self.cfg.num_steps_per_env,
                [self.dim_params["policy_dim"]],
                [self.dim_params["critic_dim"]],
                [self.env.num_actions],
                use_sapg=True,
            )
        else:
            self.alg.init_storage(
                self.env.num_envs,
                self.cfg.num_steps_per_env,
                [self.dim_params["policy_dim"]],
                [self.dim_params["critic_dim"]],
                [self.env.num_actions],
            )
        
        # Initialize batch augmenter
        if self.use_batch_augmentation:
            self.batch_augmenter = SAPGBatchAugmenter(
                expl_coef=self.expl_coef,
                off_policy_ratio=self.off_policy_ratio,
                gamma=self.alg.cfg.gamma,
                use_leader_follower=self.use_leader_follower,
            )
        
        # Initialize normalizers with extended dimensions
        self.obs_normalizer = self.cfg.obs_normalize_cfg.construct_from_cfg(
            shape=self.env.dim_params["policy_dim"]
        )
        self.critic_normalizer = self.cfg.critic_normalize_cfg.construct_from_cfg(
            shape=self.env.dim_params["critic_dim"]
        )
        self.obs_normalizer.to(self.device)
        self.critic_normalizer.to(self.device)
    
    @timeit("collection_time")
    def sample_rollout(self, ep_infos, obs, critic_obs, **kwargs):
        """
        Sample rollout with exploration coefficient embeddings.
        
        Augments observations with exploration coefficient embeddings before
        passing them to the algorithm.
        """
        rollout_datas = []
        with torch.inference_mode():
            for i in range(self.cfg.num_steps_per_env):
                # Augment observations with exploration coefficient embeddings
                obs_augmented = self.expl_coef.augment_observations(obs)
                critic_obs_augmented = self.expl_coef.augment_observations(critic_obs)
                
                # Get actions from algorithm
                actions = self.alg.act(obs_augmented, critic_obs_augmented)
                
                # Step environment
                obs, reward, done, infos = self.env.step(actions.to(self.env.device))
                critic_obs = infos["observations"].get("critic", obs)
                obs, critic_obs, reward, done = (
                    obs.to(self.device),
                    critic_obs.to(self.device),
                    reward.to(self.device),
                    done.to(self.device),
                )
                
                # Normalize observations
                obs = self.obs_normalizer(obs)
                critic_obs = self.critic_normalizer(critic_obs)
                
                # Process environment step
                self.process_env_step(reward, done, infos)
                
                # Store rollout data
                rollout_datas.append((obs, critic_obs, actions, reward, done, infos))
                
                # Update episode statistics
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
            
            # Compute returns (augment last observation first)
            last_obs_augmented = self.expl_coef.augment_observations(obs)
            last_critic_obs_augmented = self.expl_coef.augment_observations(critic_obs)
            self.alg.compute_returns(last_critic_obs_augmented)
            
            # Process rollout
            process_infos = self.process_rollout(rollout_datas)
            sample_infos = {
                "cur_reward_sum": self.cur_reward_sum,
                "cur_episode_length": self.cur_episode_length,
            }
            sample_infos.update(process_infos)
            return sample_infos, obs, critic_obs
    
    def _learn(self, num_learning_iterations: int):
        """
        Training loop with batch augmentation support.
        
        If batch augmentation is enabled, augments the batch before algorithm update.
        """
        ep_infos = []
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()
        
        for it in range(start_iter, tot_iter):
            # 1. Collect rollout
            sample_infos, obs, critic_obs = self.sample_rollout(ep_infos, obs, critic_obs)
            collection_time = sample_infos["collection_time"]
            
            # 2. Algorithm update (with SAPG batch augmentation if enabled)
            start = time.time()
            if self.use_batch_augmentation and isinstance(self.alg, SAPGPPO):
                alg_update_infos = self.alg.update(batch_augmenter=self.batch_augmenter)
            else:
                alg_update_infos = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            
            # 3. Logging and saving
            self.current_learning_iteration = it
            if self.logger.log_dir is not None:
                self.logger.log(self, locals())
            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
        
        self.save(
            os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
        )
    
    def get_inference_policy(self, device=None):
        """
        Get inference policy for deployment.
        
        For SAPG, we use the embedding from the block with lowest exploration
        (typically the last block, block_idx = num_blocks - 1), which corresponds
        to the most exploitative policy. This is the closest to the final policy
        we want to deploy.
        
        The final policy embedding is the one with the lowest exploration coefficient,
        which means it focuses on exploitation rather than exploration.
        """
        if device is not None:
            self.actor_critic.to(device)
        
        # Get embedding for the block with lowest exploration (last block)
        # This corresponds to the most exploitative policy (exploration coefficient = 0.0)
        final_block_idx = self.expl_coef.num_blocks - 1
        final_embd = self.expl_coef.get_embeddings_for_block(final_block_idx)
        
        def inference_fn(obs):
            """
            Inference function that augments observations with final embedding.
            
            Args:
                obs: Raw observations [batch_size, obs_dim]
                
            Returns:
                Actions [batch_size, action_dim]
            """
            # Normalize observations
            obs_norm = self.obs_normalizer(obs)
            
            # Augment with final embedding (lowest exploration)
            # final_embd: [num_envs, embd_dim]
            # We need to expand it to match obs batch size
            batch_size = obs_norm.shape[0]
            if batch_size == 1:
                # Single observation: use first environment's embedding
                embd = final_embd[0:1]  # [1, embd_dim]
            else:
                # Multiple observations: repeat embedding
                # For inference, we typically use the same embedding for all
                embd = final_embd[0:1].expand(batch_size, -1)  # [batch_size, embd_dim]
            
            # Concatenate observation and embedding
            obs_augmented = torch.cat([obs_norm, embd], dim=-1)
            
            # Get action
            return self.actor_critic.act_inference(obs_augmented)
        
        return inference_fn


@configclass
class SAPGOnPolicyRunnerCfg(OnPolicyRunnerCfg):
    """Configuration for SAPG on-policy runner."""
    
    class_type: type[SAPGOnPolicyRunner] = SAPGOnPolicyRunner
    
    # SAPG specific configuration
    exploration_coef_cfg: ExplorationCoefficientCfg = MISSING
    """Configuration for exploration coefficient manager."""
    
    off_policy_ratio: float = 1.0
    """Ratio of off-policy blocks to include (controls how many blocks to repeat)."""
    
    use_leader_follower: bool = False
    """Whether to use Leader-Follower mode (only keep leader block data)."""
    
    use_batch_augmentation: bool = True
    """Whether to enable batch augmentation (core SAPG feature)."""

