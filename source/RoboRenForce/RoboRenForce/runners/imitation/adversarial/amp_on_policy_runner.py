from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import MISSING

import torch

from RoboRenForce import configclass
from RoboRenForce.algorithms.imitation.adverserial.amp_ppo import AMPPPO, AMPPPOCfg
from RoboRenForce.components.actor_critic_pack import ActorCritic
from RoboRenForce.components.discriminator import Discriminator, DiscriminatorCfg
from RoboRenForce.components.normalizer import NormalizerBaseCfg
from RoboRenForce.runners.base_runner import BaseRunner, BaseRunnerCfg
from RoboRenForce.runners.logger import LoggerBaseCfg
from RoboRenForce.prototype.gym import RoboRenForceVecEnv

from RoboRenForce.utils.normalizer import RunningMeanStd
from RoboRenForce.utils.logging import timeit

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from RRF_isaaclab_tasks.env_wrapper.adversarial_wrapper import (
        AMPEnvWrapper,
        MotionDataset,
        MotionDatasetCfg,
    )

class AMPOnPolicyImitationRunner(BaseRunner):
    """On-policy runner for AMP-style adversarial PPO training."""

    env: "RoboRenForceVecEnv"
    cfg: "AMPOnPolicyImitationRunnerCfg"
    alg: AMPPPO
    actor_critic: ActorCritic

    def __init__(
        self,
        train_cfg: "AMPOnPolicyImitationRunnerCfg",
        env: "RoboRenForceVecEnv",
        log_dir=None,
        device: str = "cpu",
    ):
        self.alg_cfg: AMPPPOCfg = train_cfg.algorithm
        self.policy_cfg = train_cfg.policy
        self.amp_data_cfg: "MotionDatasetCfg" = train_cfg.amp_data
        super().__init__(train_cfg=train_cfg, env=env, log_dir=log_dir, device=device)

    def init_components(self):
        """Initialize policy, discriminator, AMP dataset and algorithm."""
        # Wrap env with AMP wrapper to expose AMP observations
        if not isinstance(self.env, AMPEnvWrapper):
            # re-wrap underlying vec env
            self.env = AMPEnvWrapper(self.env.env, clip_actions=self.env.clip_actions)

        # Dimensions
        num_actor_obs = self.env.dim_params["policy_dim"]
        num_critic_obs = self.env.dim_params.get("critic_dim", num_actor_obs)
        num_actions = self.env.num_actions

        # Policy (Actor-Critic pack)
        self.actor_critic: ActorCritic = self.policy_cfg.construct_from_cfg(
            state_dim=num_actor_obs,
            critic_dim=num_critic_obs,
            action_dim=num_actions,
            device=self.device,
        )
        self.actor_critic.to(self.device)

        # Motion dataset (expert AMP transitions)
        amp_dataset: "MotionDataset" = self.amp_data_cfg.construct_from_cfg(
            env=self.env.unwrapped, device=self.device
        )

        # AMP observation dimension
        with torch.no_grad():
            amp_obs_dim = self.env.get_amp_observations().shape[-1]

        # AMP normalizer
        self.amp_normalizer = RunningMeanStd(shape=amp_obs_dim, device=self.device)

        # Discriminator
        discr_input_dim = amp_obs_dim * 2
        discr_cfg = DiscriminatorCfg(
            backbone_cfg=self.alg_cfg.discriminator_backbone_cfg,
            amp_reward_coef=self.alg_cfg.amp_reward_coef,
            task_reward_lerp=self.alg_cfg.amp_task_reward_lerp,
        )
        self.discriminator: Discriminator = discr_cfg.construct_from_cfg(
            input_dim=discr_input_dim,
            device=self.device,
        )

        # Algorithm
        self.alg = self.alg_cfg.construct_from_cfg(
            actor_critic=self.actor_critic,
            discriminator=self.discriminator,
            amp_normalizer=self.amp_normalizer,
            motion_dataset=amp_dataset,
            amp_obs_dim=amp_obs_dim,
            device=self.device,
        )

        # Storage
        self.alg.init_storage(
            self.env.num_envs,
            self.cfg.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [num_actions],
        )

        # Observation normalizer for policy obs
        self.obs_normalizer = self.cfg.obs_normalize_cfg.construct_from_cfg(
            shape=num_actor_obs
        )
        self.obs_normalizer.to(self.device)

        # Logging buffers
        self.rewbuffer = deque(maxlen=100)
        self.ampbuffer = deque(maxlen=100)
        self.discribuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Start AMP PPO training."""
        self.logger.init_logger()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        self.cur_amp_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        self.cur_discri_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        self._learn(num_learning_iterations=num_learning_iterations)

    def _learn(self, num_learning_iterations: int):
        """Internal AMP PPO learning loop."""
        ep_infos = []
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        amp_obs = self.env.get_amp_observations()

        obs = self.obs_normalizer(obs.to(self.device))
        critic_obs = critic_obs.to(self.device)
        amp_obs = amp_obs.to(self.device)

        self.train_mode()

        for it in range(start_iter, tot_iter):
            sample_infos, obs, critic_obs, amp_obs = self.sample_rollout(
                ep_infos, obs, critic_obs, amp_obs
            )
            collection_time = sample_infos["collection_time"]

            start = time.time()
            alg_update_infos = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            self.current_learning_iteration = it

            if self.logger.log_dir is not None:
                log_locals = locals()
                log_locals.update(alg_update_infos)
                self.logger.log(self, log_locals)

            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    @timeit("collection_time")
    def sample_rollout(self, ep_infos, obs, critic_obs, amp_obs, **kwargs):
        """Collect one rollout of length num_steps_per_env."""
        rollout_datas = []

        with torch.inference_mode():
            for _ in range(self.cfg.num_steps_per_env):
                actions = self.alg.act(obs, critic_obs, amp_obs)

                (
                    obs_next,
                    critic_obs_next,
                    rewards,
                    dones,
                    infos,
                    reset_env_ids,
                    terminal_amp_states,
                ) = self.env.step(actions.to(self.env.device), not_amp=False)

                next_amp_obs = self.env.get_amp_observations()

                obs_next = obs_next.to(self.device)
                critic_obs_next = critic_obs_next.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                next_amp_obs = next_amp_obs.to(self.device)

                # Terminal AMP observations handling
                next_amp_obs_with_term = torch.clone(next_amp_obs)
                next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                # AMP reward (beyondAMP-style, computed inside AMPPPO)
                lerp_rewards, d_logits, amp_rewards = self.alg.compute_amp_reward(
                    amp_obs,
                    next_amp_obs_with_term,
                    rewards,
                )

                amp_obs = torch.clone(next_amp_obs)

                self.process_env_step(
                    lerp_rewards, dones, infos, next_amp_obs_with_term
                )

                rollout_datas.append(
                    (obs, critic_obs, actions, lerp_rewards, dones, infos)
                )

                if self.logger.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    if "log" in infos:
                        ep_infos.append(infos["log"])

                    self.cur_reward_sum += rewards
                    self.cur_amp_sum += amp_rewards.squeeze()
                    self.cur_discri_sum += d_logits.squeeze()
                    self.cur_episode_length += 1

                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(
                        self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self.ampbuffer.extend(
                        self.cur_amp_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self.discribuffer.extend(
                        self.cur_discri_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self.lenbuffer.extend(
                        self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_amp_sum[new_ids] = 0
                    self.cur_discri_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

                obs = obs_next
                critic_obs = critic_obs_next

        process_infos = self.process_rollout(rollout_datas)
        sample_infos = {
            "cur_reward_sum": self.cur_reward_sum,
            "cur_episode_length": self.cur_episode_length,
        }
        sample_infos.update(process_infos)

        return sample_infos, obs, critic_obs, amp_obs

    def process_env_step(self, rewards, dones, infos, amp_next_obs, **kwargs):
        """Forward environment feedback to underlying AMP algorithm."""
        self.alg.process_env_step(rewards, dones, infos, amp_next_obs)

    def save(self, path, infos=None):
        """Save model checkpoint for AMP training."""
        saved_dict = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "amp_normalizer": self.amp_normalizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        self.logger.save_model(saved_dict, path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        """Load model checkpoint."""
        loaded_dict = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer"])
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos", None)

    def get_inference_policy(self, device=None):
        """Return an inference policy function using normalized observations."""
        if device is not None:
            self.actor_critic.to(device)

        def _policy(obs: torch.Tensor) -> torch.Tensor:
            norm_obs = self.obs_normalizer(obs.to(self.device))
            return self.actor_critic.actor.act_inference(norm_obs)

        return _policy

    def train_mode(self):
        self.actor_critic.train()
        self.obs_normalizer.train()

    def eval_mode(self):
        self.actor_critic.eval()
        self.obs_normalizer.eval()


@configclass
class AMPOnPolicyImitationRunnerCfg(BaseRunnerCfg):
    """Configuration for AMP on-policy imitation runner."""

    class_type: type[AMPOnPolicyImitationRunner] = AMPOnPolicyImitationRunner

    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    resume: bool = False
    load_checkpoint: str = "model_.*.pt"

    # Policy and algorithm configs
    policy: object = MISSING
    algorithm: AMPPPOCfg = AMPPPOCfg()

    # AMP dataset config
    amp_data: "MotionDatasetCfg" = MISSING

    # Normalizer and logger configs
    obs_normalize_cfg: NormalizerBaseCfg = NormalizerBaseCfg()
    logger_cfg: LoggerBaseCfg = LoggerBaseCfg()

