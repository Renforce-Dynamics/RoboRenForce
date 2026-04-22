from __future__ import annotations

import os
import time
from collections import deque
from typing import List, Tuple, Dict, Union
import pickle
import torch
from dataclasses import MISSING

from RoboRenForce.utils.logging import timeit
from RoboRenForce import configclass
from RoboRenForce.components.actor_critic_pack import ActorCriticPackCfg
from RoboRenForce.components.normalizer import NormalizerBaseCfg
from RoboRenForce.components.nn_models.system_dynamics import SystemDynamicsMLPCfg
from RoboRenForce.runners.logger import LoggerBaseCfg
from RoboRenForce.runners.nn_model.nn_model_runner import (
    NNModelBasedRunner,
    NNModelBasedRunnerCfg,
)
from RoboRenForce.algorithms.on_policy.mbpo.mbpo import MBPO, MBPOCfg
from RoboRenForce.utils.template.module_base import ModuleBaseCfg
from RoboRenForce.prototype.classic import RoboRenForceVecEnv
from RoboRenForce.algorithms.nn_model_trainer.system_dynamics_trainer import (
    SystemDynamicsTrainer,
    SystemDynamicsTrainerCfg,
)

from RoboRenForce.buffer.direct_based.dynamic_replay_buffer import DynamicReplayBufferCfg

class MBPOOnPolicyRunner(NNModelBasedRunner):
    """On-policy runner with model-based rollouts (MBPO).

    This runner mirrors `OnPolicyRunner` but:
    - maintains a learned system dynamics model via `SystemDynamicsTrainer`;
    - optionally performs imagination rollouts and mixes them into PPO updates.
    """

    alg: MBPO
    cfg: "MBPOOnPolicyRunnerCfg"
    env: "lab_wrapper.RFImagineEnvWrapper"

    def __init__(
        self,
        train_cfg: "MBPOOnPolicyRunnerCfg",
        env: "lab_wrapper.RFImagineEnvWrapper",
        log_dir=None,
        device: str = "cpu",
    ):
        # Override algorithm cfg type so OnPolicyRunner sees correct subtype.
        self.alg_cfg: MBPOCfg = train_cfg.algorithm
        self.policy_cfg = train_cfg.policy
        super().__init__(train_cfg=train_cfg, env=env, log_dir=log_dir, device=device)
        
    def init_components(self):
        self.imagination_learn_flag = \
            self.cfg.imagination_num_envs > 0 and self.cfg.imagination_num_steps_per_env > 0
        num_obs, num_critic_obs = (
            self.env.dim_params["policy_dim"],
            self.env.dim_params["critic_dim"],
        )
        self.actor_critic = self.policy_cfg.construct_from_cfg(
            dim_params=self.env.dim_params
        )

        # Initialize nn model components (using base class pattern)
        self.flow_model = self.cfg.nn_model_cfg.construct_from_cfg(
            dim_params=self.env.dim_params, device=self.device
        )
        
        self.replay_buffer = self.cfg.replay_buffer_cfg.construct_from_cfg(
            self.env.dim_params, device=self.device, num_envs=self.env.num_envs
        )

        # Create nn model trainer in runner (still using SystemDynamicsTrainer)
        self.nn_model_trainer = self.cfg.nn_model_trainer_cfg.construct_from_cfg(
            replay_buffer=self.replay_buffer,
            system_dynamics=self.flow_model
        )

        self.alg = self.alg_cfg.construct_from_cfg(
            actor_critic=self.actor_critic,
            device=self.device,
        )

        self.alg.init_storage(
            self.env.num_envs,
            self.cfg.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        if self.imagination_learn_flag:
            self.alg.init_imagination_storage(
                num_envs=self.cfg.imagination_num_envs,
                num_transitions_per_env=self.cfg.imagination_num_steps_per_env,
                actor_obs_shape=[num_obs],
                critic_obs_shape=[num_critic_obs],
                action_shape=[self.env.num_actions],
            )

        # Normalisers for policy/critic inputs.
        self.obs_normalizer = self.cfg.obs_normalize_cfg.construct_from_cfg(
            shape=self.env.dim_params["policy_dim"]
        )
        self.critic_normalizer = self.cfg.critic_normalize_cfg.construct_from_cfg(
            shape=self.env.dim_params["critic_dim"]
        )
        self.obs_normalizer.to(self.device)
        self.critic_normalizer.to(self.device)
        
        self.env.set_system_dynamics(self.flow_model)

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
        self.save_example_from_replay(4, 2000)

    @torch.no_grad()
    def save_example_from_replay(
        self,
        num_examples: int = 1,
        predict_horizon: int | None = None,
        out_dir: str | None = None,
    ) -> str:
        """Sample states from the dynamics replay buffer and roll out policy + nn model.

        This utility is intended for debugging / visualization:
        - Sample initial dynamic/action histories from the replay buffer;
        - Use the current policy and system dynamics model to roll out for a
          fixed prediction horizon;
        - Record the resulting dynamics and actions into a pickle file.

        Args:
            num_examples: Number of trajectories to sample from the replay buffer.
            predict_horizon: Number of model-predicted steps to roll out.
                If None, uses ``system_dynamics_trainer_cfg.dynamic_update_forecast_horizon``.
            out_dir: Directory to save the pickle file into. If None, defaults to
                ``<log_dir>/nn_model_examples``.

        Returns:
            The path to the saved pickle file.
        """
        if predict_horizon is None:
            predict_horizon = self.cfg.nn_model_trainer_cfg.dynamic_update_forecast_horizon

        horizon = self.flow_model.history_horizon
        # Sample initial dynamic/action histories from replay buffer.
        generator = self.replay_buffer.mini_batch_generator(
            sequence_length=horizon,
            num_mini_batches=1,
            mini_batch_size=num_examples,
        )
        dynamic_batch, action_batch, _, _, _, _ = next(generator)
        # Shapes: [B, horizon, dyn_dim] / [B, horizon, act_dim]
        dynamic_history = dynamic_batch.clone()
        action_history = action_batch.clone()

        B, _, dyn_dim = dynamic_history.shape
        _, _, act_dim = action_history.shape

        pred_dynamics: List[torch.Tensor] = []
        pred_actions: List[torch.Tensor] = []

        with torch.inference_mode():
            for _ in range(predict_horizon):
                # Build imagination observation from dynamic/action history.
                imagination_obs = self.env.get_imagination_observation(dynamic_history, action_history)
                critic_obs = imagination_obs
                imagination_obs_norm = self.obs_normalizer(imagination_obs)
                critic_obs_norm = self.critic_normalizer(critic_obs)

                # Policy action in imagination.
                imagination_actions = self.alg.act_imagination(imagination_obs_norm, critic_obs_norm)

                # World model rollout (mirror RFImagineEnvWrapper.imagination_step core).
                current_dynamic = dynamic_history[:, -1:]  # [B, 1, dyn_dim]
                current_action = imagination_actions.unsqueeze(1)  # [B, 1, act_dim]
                dynamic_seq = torch.cat([dynamic_history, current_dynamic], dim=1)
                action_seq = torch.cat([action_history, current_action], dim=1)

                next_dynamic_pred, _, _, _, _ = self.flow_model(dynamic_seq, action_seq)

                # Record predictions.
                pred_dynamics.append(next_dynamic_pred.clone())
                pred_actions.append(imagination_actions.clone())

                # Update histories with predicted next state / chosen action.
                dynamic_history = torch.cat(
                    [dynamic_history[:, 1:], next_dynamic_pred.unsqueeze(1)], dim=1
                )
                action_history = torch.cat(
                    [action_history[:, 1:], current_action], dim=1
                )
        pred_dynamic_tensor = torch.stack(pred_dynamics, dim=1)  # [B, T_pred, dyn_dim]
        pred_action_tensor = torch.stack(pred_actions, dim=1)    # [B, T_pred, act_dim]
        data = {
            "seed_dynamic": dynamic_batch.detach().cpu().numpy(),
            "seed_action": action_batch.detach().cpu().numpy(),
            "pred_dynamic": pred_dynamic_tensor.detach().cpu().numpy(),
            "pred_action": pred_action_tensor.detach().cpu().numpy(),
            "dynamic_dim": dyn_dim,
            "action_dim": act_dim,
            "history_horizon": horizon,
            "predict_horizon": predict_horizon,
            "dim_params": dict(self.env.dim_params),
        }
        if out_dir is None:
            base_dir = self.logger.log_dir or "."
            out_dir = os.path.join(base_dir, "nn_model_examples")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"example_it_{self.current_learning_iteration}.pkl"
        )
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        return out_path

    def _learn(self, num_learning_iterations: int):
        ep_infos = []
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()

        for it in range(start_iter, tot_iter):
            # 1) Collect real rollouts
            sample_infos, obs, critic_obs = self.sample_rollout(
                ep_infos, obs, critic_obs, it, start_iter
            )
            collection_time = sample_infos["collection_time"]
            
            start = time.time()
            # 2) Update nn model (using base class method)
            sys_losses = self.nn_model_trainer.update_system_dynamics()
            sys_losses = {f"Model/{k}": v for k, v in sys_losses.items()}

            # 3) Policy update (with or without imagination)
            if it >= start_iter + self.cfg.nn_model_warmup_iterations:
                if self.imagination_learn_flag:
                    imagine_infos = self._imagine()
                    sample_infos.update(imagine_infos)
                    alg_update_infos = self.alg.update(use_imagination=True)
                else:
                    alg_update_infos = self.alg.update(use_imagination=False)
            elif not self.cfg.frozen_wramup:
                alg_update_infos = self.alg.update(use_imagination=False)
            else:
                alg_update_infos = {}

            alg_update_infos.update(sys_losses)
            stop = time.time()
            learn_time = stop - start

            self.current_learning_iteration = it
            if self.logger.log_dir is not None:
                # Merge infos for logging
                log_locals = locals()
                self.logger.log(self, log_locals)
            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
                self.save_example_from_replay(32, 500)
            ep_infos.clear()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    @timeit("collection_time")
    def sample_rollout(self, ep_infos, obs, critic_obs, it, start_iter):
        """Extended rollout sampling that also fills system-dynamics replay."""
        rollout_datas = []
        with torch.inference_mode():
            for i in range(self.cfg.num_steps_per_env):
                actions = self.alg.act(obs, critic_obs)
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

                (
                    system_dynamic,
                    system_action,
                    system_extension,
                    system_contact,
                    system_termination,
                ) = self.env.get_system_observation()
                system_reward = reward.clone()

                if not self.cfg.frozen_wramup or it >= start_iter + self.cfg.nn_model_warmup_iterations:
                    self.process_env_step(reward, done, infos)

                # Store system observation data for replay buffer
                system_data = (
                    system_dynamic,
                    system_action,
                    system_extension,
                    system_contact,
                    system_termination,
                    system_reward,
                )
                rollout_datas.append((obs, critic_obs, actions, reward, done, infos, system_data))

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

            if not self.cfg.frozen_wramup or it >= start_iter + self.cfg.nn_model_warmup_iterations:
                self.alg.compute_returns(critic_obs)

            process_infos = self.process_rollout(rollout_datas)
            sample_infos = {
                "cur_reward_sum": self.cur_reward_sum,
                "cur_episode_length": self.cur_episode_length,
            }
            sample_infos.update(process_infos)
            return sample_infos, obs, critic_obs

    def process_env_step(self, reward, done, infos, **kwargs):
        self.alg.process_env_step(reward, done, infos)

    def process_rollout(self, rollout_datas: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], Tuple]]):
        """Process rollout data and add system dynamics data to replay buffer.
        Args:
            rollout_datas: List of tuples, each containing:
                - obs: normalized policy observations
                - critic_obs: normalized critic observations
                - actions: actions taken
                - reward: rewards received
                - done: done flags
                - infos: additional information dict
                - system_data: tuple of (system_dynamic, system_action, system_extension, 
                                     system_contact, system_termination, system_reward)
        Returns:
            Dictionary with rollout statistics from parent class
        """
        # Extract system dynamics data from rollout_datas
        system_dynamics_list = []
        system_actions_list = []
        system_extensions_list = []
        system_contacts_list = []
        system_terminations_list = []
        system_rewards_list = []
        
        for rollout_data in rollout_datas:
            # Unpack rollout data (last element is system_data tuple)
            *_, system_data = rollout_data
            system_dynamic, system_action, system_extension, system_contact, system_termination, system_reward = system_data
            
            system_dynamics_list.append(system_dynamic)
            system_actions_list.append(system_action)
            if system_extension is not None:
                system_extensions_list.append(system_extension)
            if system_contact is not None:
                system_contacts_list.append(system_contact)
            if system_termination is not None:
                system_terminations_list.append(system_termination)
            system_rewards_list.append(system_reward)
        
        # Stack all data along time dimension: [num_envs, num_steps, dim]
        system_dynamics = torch.stack(system_dynamics_list, dim=1)  # [num_envs, num_steps, dynamic_dim]
        system_actions = torch.stack(system_actions_list, dim=1)  # [num_envs, num_steps, action_dim]
        system_rewards = torch.stack(system_rewards_list, dim=1).unsqueeze(-1)  # [num_envs, num_steps, 1]
        
        # Handle optional fields (only stack if all elements are not None)
        system_extension = None
        if len(system_extensions_list) > 0 and all(x is not None for x in system_extensions_list):
            system_extension = torch.stack(system_extensions_list, dim=1)
        
        system_contact = None
        if len(system_contacts_list) > 0 and all(x is not None for x in system_contacts_list):
            system_contact = torch.stack(system_contacts_list, dim=1)
        
        system_termination = None
        if len(system_terminations_list) > 0 and all(x is not None for x in system_terminations_list):
            system_termination = torch.stack(system_terminations_list, dim=1)
        
        # Add to replay buffer
        self.replay_buffer.insert(
            dynamic=system_dynamics,
            action=system_actions,
            extension=system_extension,
            contact=system_contact,
            termination=system_termination,
            reward=system_rewards,
        )
        
        # Remove system_data from rollout_datas before passing to parent
        # (parent expects original format)
        rollout_datas_for_parent = [
            (obs, critic_obs, actions, reward, done, infos)
            for obs, critic_obs, actions, reward, done, infos, _ in rollout_datas
        ]
        return super().process_rollout(rollout_datas_for_parent)

    @timeit("imagination_collection_time")
    def _imagine(self):
        """Generate imagination rollouts using system dynamics model.
        
        This method generates virtual trajectories by:
        1. Getting observations from dynamic/action history
        2. Using current policy to select actions
        3. Using system dynamics to predict next dynamic states and rewards
        4. Storing results in imagination_storage
        """

        # Clear imagination storage for new rollouts
        if self.alg.imagination_storage is not None:
            self.alg.imagination_storage.clear()

        dynamic_history, action_history = self._prepare_imagination()

        with torch.inference_mode():
            for i in range(self.cfg.imagination_num_steps_per_env):
                imagination_obs = self.env.get_imagination_observation(dynamic_history, action_history)
                critic_obs = imagination_obs  # Assume same for now, can be extended
                imagination_obs = self.obs_normalizer(imagination_obs)
                critic_obs = self.critic_normalizer(critic_obs)
                imagination_actions = self.alg.act_imagination(imagination_obs, critic_obs)
                (
                    imagination_obs_next,
                    imagination_rewards,
                    imagination_dones,
                    imagination_extras,
                    dynamic_history,
                    action_history,
                ) = self.env.imagination_step(
                    imagination_actions, dynamic_history, action_history
                )
                
                reset_env_ids = (imagination_dones.squeeze(-1) > 0).nonzero(as_tuple=False).squeeze(-1)
                if reset_env_ids.dim() == 0: reset_env_ids = reset_env_ids.unsqueeze(0)
                if len(reset_env_ids) > 0:
                    num_reset = len(reset_env_ids)
                    imagination_dynamic_history, imagination_action_history = self._prepare_imagination()
                    dynamic_history[reset_env_ids] = imagination_dynamic_history[:num_reset].clone()
                    action_history[reset_env_ids] = imagination_action_history[:num_reset].clone()
                
                # Normalize next observation
                imagination_obs_next_norm = self.obs_normalizer(imagination_obs_next)
                critic_obs_next = imagination_extras.get("observations", {}).get("critic", imagination_obs_next)
                critic_obs_next_norm = self.critic_normalizer(critic_obs_next)

                self.alg.process_env_step(
                    imagination_rewards,
                    imagination_dones,
                    imagination_extras,
                    imagination=True,
                )

                imagination_obs = imagination_obs_next_norm
                critic_obs = critic_obs_next_norm

            if self.alg.imagination_storage is not None:
                self.alg.compute_imagination_returns(critic_obs)

    @torch.no_grad()
    def _prepare_imagination(self):
        """Prepare initial dynamic/action history for imagination rollouts.
        
        Samples initial dynamic and action histories from the system replay buffer.
        
        Returns:
            dynamic_history: [num_imagination_envs, history_horizon, dynamic_dim]
            action_history: [num_imagination_envs, history_horizon, action_dim]
        """
        horizon = self.flow_model.history_horizon
        imagination_generator = self.replay_buffer.mini_batch_generator(
            sequence_length=horizon,
            num_mini_batches=1,
            mini_batch_size=self.alg.imagination_storage.num_envs,
        )
        imagination_dynamic_history, imagination_action_history = next(imagination_generator)[:2]
        return imagination_dynamic_history, imagination_action_history

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.actor_critic.state_dict(),
            "obs_norm_state_dict": self.obs_normalizer.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "nn_model_state_dict": self.flow_model.state_dict(),
            "nn_model_optimizer_state_dict": self.nn_model_trainer.model_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        self.logger.save_model(saved_dict, path, self.current_learning_iteration)

@configclass
class MBPOOnPolicyRunnerCfg(NNModelBasedRunnerCfg):
    class_type: type[MBPOOnPolicyRunner] = MBPOOnPolicyRunner

    # Algorithm and policy
    policy          : ActorCriticPackCfg = MISSING
    algorithm       : MBPOCfg = MISSING
    frozen_wramup   : bool = True

    # World model configuration (uses SystemDynamicsMLPCfg under the hood)
    nn_model_cfg: "SystemDynamicsMLPCfg" = MISSING
    replay_buffer_cfg: DynamicReplayBufferCfg = DynamicReplayBufferCfg(buffer_size=320)
    nn_model_trainer_cfg: SystemDynamicsTrainerCfg = SystemDynamicsTrainerCfg()
    nn_model_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()  # kept for backward compatibility
    nn_model_action_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()
    nn_model_warmup_iterations: int = 0
    imagination_num_envs: int = 0
    imagination_num_steps_per_env: int = 0
