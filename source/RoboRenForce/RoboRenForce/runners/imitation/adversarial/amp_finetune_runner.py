from __future__ import annotations

import copy
import os
import time
from collections import deque

import torch

from RoboRenForce import configclass
from RoboRenForce.algorithms.imitation.adverserial.amp_ppo import AMPPPO, AMPPPOCfg
from RoboRenForce.components.actor_critic_pack import ActorCritic
from RoboRenForce.components.normalizer import NormalizerBaseCfg
from RoboRenForce.runners.imitation.adversarial.amp_on_policy_runner import (
    AMPOnPolicyImitationRunner,
    AMPOnPolicyImitationRunnerCfg,
)
from RoboRenForce.runners.logger import LoggerBaseCfg
from RoboRenForce.prototype.classic import RoboRenForceVecEnv


class AMPFinetuneImitationRunner(AMPOnPolicyImitationRunner):
    """AMP finetuning runner with residual KL reward to previous policy."""

    env: "RoboRenForceVecEnv"
    cfg: "AMPFinetuneImitationRunnerCfg"
    alg: AMPPPO
    actor_critic: ActorCritic

    def __init__(
        self,
        train_cfg: "AMPFinetuneImitationRunnerCfg",
        env: "RoboRenForceVecEnv",
        log_dir=None,
        device: str = "cpu",
    ):
        super().__init__(train_cfg=train_cfg, env=env, log_dir=log_dir, device=device)
        self.prev_policy: ActorCritic | None = None

    def compute_policy_kl(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute per-env KL(pi_current || pi_prev) as residual penalty."""
        if self.prev_policy is None:
            return torch.zeros(obs.shape[0], device=self.device)

        with torch.no_grad():
            self.actor_critic.actor.act(obs)
            self.prev_policy.actor.act(obs)
            cur_dist = self.actor_critic.actor.distribution
            prev_dist = self.prev_policy.actor.distribution
            kl = torch.distributions.kl_divergence(cur_dist, prev_dist)
            kl = kl.sum(dim=-1)
        return kl

    def _learn(self, num_learning_iterations: int):
        """Internal learning loop with residual KL reward."""
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

        resibuffer = deque(maxlen=100)
        cur_resi_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        for it in range(start_iter, tot_iter):
            start = time.time()
            rollout_datas = []

            with torch.inference_mode():
                for _ in range(self.cfg.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    (
                        obs_next,
                        critic_obs_next,
                        task_rewards,
                        dones,
                        infos,
                        reset_env_ids,
                        terminal_amp_states,
                    ) = self.env.step(actions.to(self.env.device), not_amp=False)

                    next_amp_obs = self.env.get_amp_observations()

                    obs_next = obs_next.to(self.device)
                    critic_obs_next = critic_obs_next.to(self.device)
                    task_rewards = task_rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)

                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    lerp_rewards, d_logits, amp_rewards = (
                        self.alg.discriminator.predict_amp_reward(
                            amp_obs,
                            next_amp_obs_with_term,
                            task_rewards,
                            normalizer=self.alg.amp_normalizer,
                        )
                    )

                    if self.prev_policy is not None:
                        kl = self.compute_policy_kl(obs)
                        kl_reward = -kl
                        beta = self.cfg.kl_coef
                        lerp_rewards = lerp_rewards + beta * kl_reward
                    else:
                        kl_reward = torch.zeros_like(task_rewards)

                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(
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

                        self.cur_reward_sum += task_rewards
                        self.cur_amp_sum += amp_rewards.squeeze()
                        self.cur_discri_sum += d_logits.squeeze()
                        self.cur_episode_length += 1
                        cur_resi_sum += kl_reward

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
                        resibuffer.extend(
                            cur_resi_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        self.lenbuffer.extend(
                            self.cur_episode_length[new_ids][:, 0]
                            .cpu()
                            .numpy()
                            .tolist()
                        )

                        self.cur_reward_sum[new_ids] = 0
                        self.cur_amp_sum[new_ids] = 0
                        self.cur_discri_sum[new_ids] = 0
                        cur_resi_sum[new_ids] = 0
                        self.cur_episode_length[new_ids] = 0

                    obs = obs_next
                    critic_obs = critic_obs_next

            collection_time = time.time() - start

            start = time.time()
            self.alg.compute_returns(critic_obs)
            alg_update_infos = self.alg.update()
            learn_time = time.time() - start

            self.current_learning_iteration = it

            if self.logger.log_dir is not None:
                log_locals = locals()
                log_locals.update(alg_update_infos)
                log_locals["resibuffer"] = resibuffer
                self.logger.log(self, log_locals)

            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def load(self, path, load_optimizer: bool = False):
        """Load checkpoint and freeze as previous policy for residual KL."""
        loaded_dict = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(
                loaded_dict["discriminator_state_dict"]
            )
            self.alg.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer"])
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

        self.prev_policy = copy.deepcopy(self.actor_critic)
        self.prev_policy.eval()
        for p in self.prev_policy.parameters():
            p.requires_grad_(False)

        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos", None)


@configclass
class AMPFinetuneImitationRunnerCfg(AMPOnPolicyImitationRunnerCfg):
    """Configuration for AMP finetuning runner with residual KL reward."""

    class_type: type[AMPFinetuneImitationRunner] = AMPFinetuneImitationRunner

    kl_coef: float = 1e-3

