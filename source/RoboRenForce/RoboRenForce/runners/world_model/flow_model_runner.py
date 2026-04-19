from __future__ import annotations

import os
import time
from collections import deque
from typing import List, Dict
import torch
from dataclasses import MISSING

from RoboRenForce import configclass
from RoboRenForce.utils.logging import timeit
from RoboRenForce.prototype.classic import RoboRenForceVecEnv
from RoboRenForce.algorithms.on_policy.ppo import PPO, PPOCfg
from RoboRenForce.runners.world_model.world_model_runner import (
    WorldModelBasedRunner,
    WorldModelBasedRunnerCfg,
)

class FlowModelRunner(WorldModelBasedRunner):
    """On-policy runner that trains belief flow models as world models.

    This runner:
    - uses PPO for policy learning;
    - maintains and trains belief_flow and belief_obs_update models via `FlowModelTrainer`;
    - collects belief-related data from rollouts (low_obs, high_obs, belief states);
    - optionally maintains system dynamics model for imagination rollouts.
    """

    alg: "PPO"
    cfg: "FlowModelRunnerCfg"
    env: "lab_wrapper.RFSensorEnvWrapper"

    def __init__(
        self,
        train_cfg: "FlowModelRunnerCfg",
        env: "lab_wrapper.RFSensorEnvWrapper",
        log_dir=None,
        device: str = "cpu",
    ):
        # Override algorithm cfg type so OnPolicyRunner sees correct subtype.
        self.alg_cfg: PPOCfg = train_cfg.algorithm
        self.policy_cfg = train_cfg.policy
        super().__init__(train_cfg=train_cfg, env=env, log_dir=log_dir, device=device)
        
    def init_components(self):
        # No imagination rollout in this runner; PPO is purely on real env rollouts.
        obs_policy_dim, obs_critic_dim, obs_slow_dim, action_dim = (
            self.env.dim_params["policy_dim"],
            self.env.dim_params["critic_dim"],
            self.env.dim_params["slow_dim"],
            self.env.dim_params["action_dim"],
        )
        belief_dim = self.cfg.belief_dim

        # Initialize actor-critic with built-in belief encoder
        dim_params_with_belief = dict(self.env.dim_params)
        dim_params_with_belief["belief_dim"] = belief_dim
        self.actor_critic = self.policy_cfg.construct_from_cfg(
            dim_params=dim_params_with_belief
        )
        self.belief_encoder = self.actor_critic.actor.encoder
        
        self.belief_obs_update = self.cfg.belief_updater_cfg.construct_from_cfg(
            dim_params=self.env.dim_params, device=self.device
        )
        self.flow_model = self.cfg.world_model_cfg.construct_from_cfg(
            dim_params=self.env.dim_params, device=self.device
        )
        
        # Initialize belief-aware storage
        self.storage = BeliefRolloutStorage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.cfg.num_steps_per_env,
            obs_shape=[obs_policy_dim],
            privileged_obs_shape=[obs_critic_dim],
            actions_shape=[action_dim],
            belief_dim=belief_dim,
            slow_obs_shape=[obs_slow_dim],
            device=self.device,
        )
        
        # Initialize algorithm with pre-constructed storage
        self.alg = self.alg_cfg.construct_from_cfg(
            actor_critic=self.actor_critic, device=self.device,
        )
        self.alg.init_storage(storage=self.storage)
        
        # Initialize flow model trainer
        self.flow_model_trainer = self.cfg.flow_model_trainer_cfg.construct_from_cfg(
            replay_buffer=self.storage,
            belief_flow=self.flow_model,
            belief_obs_update=self.belief_obs_update,
        )

        self.obs_normalizer = self.cfg.obs_normalize_cfg.construct_from_cfg(
            shape=self.env.dim_params["policy_dim"]
        )
        self.critic_normalizer = self.cfg.critic_normalize_cfg.construct_from_cfg(
            shape=self.env.dim_params["critic_dim"]
        )
        self.obs_normalizer.to(self.device)
        self.critic_normalizer.to(self.device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        self.logger.init_logger()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
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
        ep_infos = []
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        obs, extras = self.env.get_observations()
        extra_obs = extras["observations"]
        critic_obs = extra_obs.get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()

        for it in range(start_iter, tot_iter):
            # 1) Collect real rollouts
            sample_infos, obs, critic_obs, extra_obs = self.sample_rollout(
                ep_infos, obs, critic_obs, extra_obs
            )
            
            start = time.time()
            # 2) Update flow models
            flow_losses = self.flow_model_trainer.update_flow_models()
            flow_losses = {f"Flow/{k}": v for k, v in flow_losses.items()}

            # 3) PPO policy update (real rollouts only)
            alg_update_infos = self.alg.update()

            alg_update_infos.update(flow_losses)
            stop = time.time()
            
            collection_time = sample_infos["collection_time"]
            learn_time = stop - start

            self.current_learning_iteration = it
            if self.logger.log_dir is not None:
                log_locals = locals()
                self.logger.log(self, log_locals)
            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    @timeit("collection_time")
    def sample_rollout(self, ep_infos, obs, critic_obs, extra_obs):
        """Rollout sampling that also fills flow-model replay buffers with belief states."""
        rollout_datas = []
        with torch.inference_mode():
            for i in range(self.cfg.num_steps_per_env):
                actions = self.alg.act(obs, critic_obs, slow_obs=extra_obs["slow"])
                self.alg.transition.belief = self.actor_critic.actor.belief
                self.alg.transition.slow_obs = extra_obs["slow"]
                
                obs, reward, done, infos = self.env.step(actions.to(self.env.device))
                critic_obs = infos["observations"].get("critic", obs)
                obs, critic_obs, reward, done = (
                    obs.to(self.device),
                    critic_obs.to(self.device),
                    reward.to(self.device),
                    done.to(self.device),
                )
                obs = self.obs_normalizer(obs)
                critic_obs = self.critic_normalizer(critic_obs)

                self.process_env_step(reward, done, infos)
                rollout_datas.append((obs, critic_obs, actions, reward, done, infos))

                if self.logger.log_dir is not None:
                    if "episode" in infos   : ep_infos.append(infos["episode"])
                    elif "log" in infos     : ep_infos.append(infos["log"])
                    self.cur_reward_sum += reward
                    self.cur_episode_length += 1
                    new_ids = (done > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.alg.compute_returns(critic_obs)

            process_infos = self.process_rollout(rollout_datas)
            sample_infos = {
                "cur_reward_sum": self.cur_reward_sum,
                "cur_episode_length": self.cur_episode_length,
            }
            sample_infos.update(process_infos)
            return sample_infos, obs, critic_obs, infos["observations"]

    def process_env_step(self, reward, done, infos, **kwargs):
        self.alg.process_env_step(reward, done, infos)
        # self.belief_encoder.reset(done)

    def save(self, path, infos=None):
        ...

from RoboRenForce.algorithms.world_model_trainer.flow_model_trainer import FlowModelTrainerCfg
from RoboRenForce.buffer.online_rollout.belief_rollout_storage import BeliefRolloutStorage
from RoboRenForce import components, networks

@configclass
class FlowModelRunnerCfg(WorldModelBasedRunnerCfg):
    class_type: type[FlowModelRunner] = FlowModelRunner

    # Policy with built-in belief encoder
    policy: components.ActorCriticPackCfg = components.ActorCriticPackCfg(
        actor_cfg=components.BeliefEncoderActorCfg(
            encoder_cfg=components.VecStateEncoderCfg(
                backbone_cfg=networks.MLPCfg(
                    hidden_features=[256, 256],
                    activations=[
                        [('ReLU', {})],
                        [('ReLU', {})],
                        [],
                    ]
                )
            ),
            actor_cfg=components.StateIndStdActorCfg(
                backbone_cfg=networks.MLPCfg(
                    hidden_features=[512, 256, 128],
                    activations=[
                        [('ELU', {})],
                        [('ELU', {})],
                        [('ELU', {})],
                        [],
                    ]
                ),
                use_log_std=False,
            ),
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[
                    [('ELU', {})],
                    [('ELU', {})],
                    [('ELU', {})],
                    [],
                ]
            )
        ),
    )
    
    # Algorithm
    algorithm: PPOCfg = PPOCfg()

    # World model configuration (belief flow)
    world_model_cfg: components.world_models.BeliefFlowModelCfg = MISSING
    
    # Belief observation updater
    belief_updater_cfg: components.world_models.BeliefUpdaterCfg = MISSING
    
    # Dimension of belief hidden state
    belief_dim: int = 64
    
    # Flow model trainer configuration
    flow_model_trainer_cfg: FlowModelTrainerCfg = FlowModelTrainerCfg()
