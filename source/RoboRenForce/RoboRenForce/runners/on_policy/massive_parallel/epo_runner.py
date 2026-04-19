from __future__ import annotations

import os
import time
from dataclasses import MISSING

import torch

from RoboRenForce import configclass
from RoboRenForce.algorithms import AlgorithmBaseCfg
from RoboRenForce.components.actor_critic_pack import ActorCriticPackCfg
from RoboRenForce.components.normalizer import NormalizerBaseCfg
from RoboRenForce.runners.logger import LoggerBaseCfg

from RoboRenForce.prototype.classic import RoboRenForceVecEnv
from RoboRenForce.utils.logging import timeit

from RoboRenForce.algorithms.on_policy.epo.epo_observer import EPOObserver
from RoboRenForce.algorithms.on_policy.epo.exploration_coefficient_epo import EPOExplorationCoefficientCfg
from RoboRenForce.algorithms.on_policy.sapg.sapg_ppo import SAPGPPO
from RoboRenForce.algorithms.on_policy.sapg.sapg_augmentation import SAPGBatchAugmenter

from .sapg_runner import SAPGOnPolicyRunner, SAPGOnPolicyRunnerCfg

class EPOOnPolicyRunner(SAPGOnPolicyRunner):
    """
    EPO-enabled on-policy runner.

    Extends SAPGOnPolicyRunner by:
    - Using EPOExplorationCoefficient (with merge_block_params);
    - Attaching EPOObserver to evolve block-level parameters during training.
    """

    def __init__(
        self,
        cfg: "EPOOnPolicyRunnerCfg",
        env: "RoboRenForceVecEnv",
        log_dir=None,
        device="cpu",
    ):
        # EPO-specific options
        self.epo_interval_steps = cfg.epo_interval_steps
        self.epo_warmup_steps = cfg.epo_warmup_steps
        super().__init__(cfg=cfg, env=env, log_dir=log_dir, device=device)

    def init_components(self):
        """
        Same as SAPGOnPolicyRunner, but uses EPOExplorationCoefficient and
        initializes an EPOObserver.
        """
        # Let SAPG initialize actor_critic / alg / normalizers first.
        # Ensure exploration_coef_cfg class_type is EPOExplorationCoefficient.
        if isinstance(self.expl_coef_cfg, EPOExplorationCoefficientCfg):
            pass
        # 调用父类 init_components 以完成标准 SAPG 初始化
        super().init_components()

        # Replace expl_coef with EPOExplorationCoefficient if needed
        # (super().init_components already created self.expl_coef)
        if not isinstance(self.expl_coef, type(EPOObserver.__init__.__annotations__["expl_coef"])):
            # 重新构造为 EPOExplorationCoefficient
            self.expl_coef = self.expl_coef_cfg.construct_from_cfg(
                num_envs=self.env.num_envs,
                device=self.device,
            )

        # Create batch augmenter (same as SAPG)
        if isinstance(self.alg, SAPGPPO):
            self.batch_augmenter = SAPGBatchAugmenter(
                expl_coef=self.expl_coef,
                off_policy_ratio=self.off_policy_ratio,
                gamma=self.alg.cfg.gamma,
                use_leader_follower=self.use_leader_follower,
            )

        # Initialize EPO observer
        self.epo_observer = EPOObserver(
            expl_coef=self.expl_coef,
            num_envs=self.env.num_envs,
            interval_steps=self.epo_interval_steps,
            warmup_steps=self.epo_warmup_steps,
            best_embeddings=True,
        )

    @timeit("collection_time")
    def sample_rollout(self, ep_infos, obs, critic_obs, **kwargs):
        """
        Same as SAPG sample_rollout, but calls EPOObserver.process_infos at each step.
        """
        rollout_datas = []
        with torch.inference_mode():
            for i in range(self.cfg.num_steps_per_env):
                obs_augmented = self.expl_coef.augment_observations(obs)
                critic_obs_augmented = self.expl_coef.augment_observations(critic_obs)

                actions = self.alg.act(obs_augmented, critic_obs_augmented)

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

                # Update episode statistics before EPO observer (so cur_reward_sum includes current step)
                if self.logger.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])
                    self.cur_reward_sum += reward
                    self.cur_episode_length += 1

                # EPO observer: consume infos for block-level objectives
                # Use true_objective if available, otherwise fallback to cumulative reward
                done_indices = (done > 0).nonzero(as_tuple=False)
                if done_indices.numel() > 0:
                    frame = getattr(self.alg, "frame", 0)
                    self.epo_observer.update_frame(frame)
                    # Pass cumulative reward for done episodes as fallback
                    # cur_reward_sum now contains cumulative reward including current step
                    cumulative_rewards = self.cur_reward_sum if hasattr(self, 'cur_reward_sum') else None
                    self.epo_observer.process_infos(infos, done_indices, rewards=cumulative_rewards)

                rollout_datas.append((obs, critic_obs, actions, reward, done, infos))

                # Reset episode statistics after EPO observer
                if self.logger.log_dir is not None:
                    new_ids = (done > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(
                        self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self.lenbuffer.extend(
                        self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            last_obs_augmented = self.expl_coef.augment_observations(obs)
            last_critic_obs_augmented = self.expl_coef.augment_observations(critic_obs)
            self.alg.compute_returns(last_critic_obs_augmented)

            process_infos = self.process_rollout(rollout_datas)
            sample_infos = {
                "cur_reward_sum": self.cur_reward_sum,
                "cur_episode_length": self.cur_episode_length,
            }
            sample_infos.update(process_infos)
            return sample_infos, obs, critic_obs

    def _learn(self, num_learning_iterations: int):
        """
        Training loop: same as SAPG, but calls EPOObserver.after_steps each iteration.
        """
        ep_infos = []
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()

        for it in range(start_iter, tot_iter):
            sample_infos, obs, critic_obs = self.sample_rollout(ep_infos, obs, critic_obs)

            start = time.time()
            if isinstance(self.alg, SAPGPPO):
                alg_update_infos = self.alg.update(batch_augmenter=self.batch_augmenter)
            else:
                alg_update_infos = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            # EPO evolution step
            frame = getattr(self.alg, "frame", 0)
            self.epo_observer.update_frame(frame)
            self.epo_observer.after_steps()

            self.current_learning_iteration = it
            if self.logger.log_dir is not None:
                self.logger.log(self, locals())
            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.save(
            os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
        )


@configclass
class EPOOnPolicyRunnerCfg(SAPGOnPolicyRunnerCfg):
    """
    Runner config for EPOOnPolicyRunner.

    Inherits from SAPGOnPolicyRunnerCfg, adds EPO-specific options, and
    defaults exploration_coef_cfg to EPOExplorationCoefficientCfg.
    """

    class_type: type[EPOOnPolicyRunner] = EPOOnPolicyRunner

    exploration_coef_cfg: EPOExplorationCoefficientCfg = EPOExplorationCoefficientCfg()

    epo_interval_steps: int = 20_000_000
    epo_warmup_steps: int = 200_000_000

