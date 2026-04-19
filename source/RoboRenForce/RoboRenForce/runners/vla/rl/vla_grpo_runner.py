"""
VLA GRPO Runner — RL fine-tuning of VLA policies with Group Relative Policy Optimization.

Training loop:
    for each iteration:
        1. Collect group_size rollouts per env using current policy
        2. Compute episode rewards (binary success from sim)
        3. Compute group-relative advantages
        4. For update_epochs:
             a. Re-evaluate logprobs under current policy
             b. Compute clipped surrogate loss with GRPO advantages
             c. Update policy

This runner works with any BasePolicy that supports:
    - predict_action(obs) → actions
    - forward(ForwardType.PPO, obs=obs, actions=actions) → {logprobs, ...}

Reference: RLinf rlinf/runners/embodied_runner.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.prototype.embodied import BasePolicy, ForwardType, EmbodiedEnv
from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithm, GRPOAlgorithmCfg


@dataclass
class RolloutData:
    """Collected rollout data for one iteration."""
    obs: list[dict[str, Any]]           # [total_rollouts] obs dicts
    actions: torch.Tensor               # [total_rollouts, action_dim]
    logprobs: torch.Tensor              # [total_rollouts]
    rewards: torch.Tensor               # [total_rollouts]
    dones: torch.Tensor                 # [total_rollouts]
    episode_returns: torch.Tensor       # [total_rollouts] cumulative rewards


class VLAGRPORunner(ModuleBase):
    """VLA RL fine-tuning runner using GRPO.

    Usage:
        runner = VLAGRPORunner(cfg, env, policy, device="cuda:0")
        runner.learn(num_iterations=100)

    The runner handles:
    - Rollout collection with grouped episodes
    - GRPO advantage computation
    - Multi-epoch policy updates with clipped surrogate
    - Checkpointing and logging
    """

    def __init__(
        self,
        cfg: "VLAGRPORunnerCfg",
        env: EmbodiedEnv,
        policy: BasePolicy,
        device: str = "cpu",
        log_dir: str = "logs/grpo",
    ):
        super().__init__()
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Algorithm
        self.grpo = GRPOAlgorithm(cfg.grpo_cfg)

        # Optimizer (only trainable params)
        self.optimizer = torch.optim.AdamW(
            list(policy.trainable_parameters()),
            lr=cfg.grpo_cfg.learning_rate,
            weight_decay=cfg.grpo_cfg.weight_decay,
        )

        # State
        self.global_step = 0
        self.total_episodes = 0
        self._writer = None

    @property
    def writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                self._writer = _DummyWriter()
        return self._writer

    @torch.no_grad()
    def collect_rollouts(self) -> RolloutData:
        """Collect group_size rollouts per env.

        Each env runs group_size episodes. Episodes are collected sequentially
        within each env (reset after done).

        Returns:
            RolloutData with [num_envs * group_size] entries per episode
        """
        self.policy.eval()
        num_envs = self.env.num_envs
        group_size = self.cfg.grpo_cfg.group_size
        max_steps = self.env.max_episode_length

        all_obs = []
        all_actions = []
        all_logprobs = []
        all_rewards = []
        all_dones = []
        all_episode_returns = []

        for g in range(group_size):
            obs, info = self.env.reset()

            episode_obs = []
            episode_actions = []
            episode_logprobs = []
            episode_rewards = torch.zeros(num_envs, device=self.device)
            done_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

            for t in range(max_steps):
                # Get action + logprob from policy
                action_result = self._get_action_with_logprob(obs)
                actions = action_result["actions"]
                logprobs = action_result["logprobs"]

                episode_obs.append(obs)
                episode_actions.append(actions)
                episode_logprobs.append(logprobs)

                # Step env
                next_obs, rewards, dones, extras = self.env.step(actions)

                # Accumulate rewards for active episodes
                episode_rewards += rewards * (~done_mask).float()
                done_mask = done_mask | dones

                obs = next_obs

                # Early termination if all envs are done
                if done_mask.all():
                    break

            # Store per-episode aggregates
            # Use last obs and last action as representative for the episode
            # (for logprob re-evaluation during training)
            last_obs = episode_obs[-1] if episode_obs else obs
            last_actions = episode_actions[-1] if episode_actions else torch.zeros(
                num_envs, self.env.num_actions, device=self.device)
            last_logprobs = episode_logprobs[-1] if episode_logprobs else torch.zeros(
                num_envs, device=self.device)

            all_obs.append(last_obs)
            all_actions.append(last_actions)
            all_logprobs.append(last_logprobs)
            all_rewards.append(episode_rewards)
            all_dones.append(done_mask)
            all_episode_returns.append(episode_rewards.clone())

        # Stack: [group_size, num_envs] → [num_envs * group_size]
        # Interleave so that groups of group_size are contiguous
        stacked_actions = torch.stack(all_actions, dim=1).reshape(-1, all_actions[0].shape[-1])
        stacked_logprobs = torch.stack(all_logprobs, dim=1).reshape(-1)
        stacked_rewards = torch.stack(all_rewards, dim=1).reshape(-1)
        stacked_dones = torch.stack(all_dones, dim=1).reshape(-1)
        stacked_returns = torch.stack(all_episode_returns, dim=1).reshape(-1)

        # Interleave obs dicts similarly
        interleaved_obs = []
        for env_idx in range(num_envs):
            for g in range(group_size):
                interleaved_obs.append(
                    {k: v[env_idx:env_idx+1] if isinstance(v, torch.Tensor) else [v[env_idx]] if isinstance(v, list) else v
                     for k, v in all_obs[g].items()}
                )

        self.total_episodes += num_envs * group_size

        return RolloutData(
            obs=interleaved_obs,
            actions=stacked_actions,
            logprobs=stacked_logprobs,
            rewards=stacked_rewards,
            dones=stacked_dones,
            episode_returns=stacked_returns,
        )

    @torch.no_grad()
    def _get_action_with_logprob(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Get actions and their log-probabilities from the policy.

        For policies that support PPO forward (returns logprobs), use that.
        Otherwise, use predict_action and estimate logprobs from Gaussian assumption.
        """
        # Try PPO forward first
        try:
            result = self.policy.forward(ForwardType.PPO, obs=obs, actions=None)
            if "logprobs" in result and "pred_actions" in result:
                return {
                    "actions": result["pred_actions"].squeeze(1) if result["pred_actions"].dim() == 3
                              and result["pred_actions"].shape[1] == 1
                              else result["pred_actions"],
                    "logprobs": result["logprobs"],
                }
        except (NotImplementedError, TypeError):
            pass

        # Fallback: predict_action + Gaussian logprob estimate
        actions = self.policy.predict_action(obs)
        if actions.dim() == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)

        # Estimate logprob under unit Gaussian centered at predicted action
        # (This is a rough approximation; proper implementation needs action distribution)
        logprobs = -0.5 * (actions ** 2).sum(dim=-1)

        return {"actions": actions, "logprobs": logprobs}

    def _recompute_logprobs(
        self,
        obs_batch: list[dict[str, Any]],
        actions_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Re-evaluate logprobs under current policy for a batch.

        Args:
            obs_batch: list of obs dicts (one per sample)
            actions_batch: [B, action_dim]

        Returns:
            logprobs: [B]
        """
        # Batch observations
        batched_obs = _batch_obs_list(obs_batch, self.device)

        try:
            result = self.policy.forward(
                ForwardType.PPO,
                obs=batched_obs,
                actions=actions_batch,
            )
            return result["logprobs"]
        except (NotImplementedError, TypeError):
            pass

        # Fallback: re-predict and compute Gaussian logprob
        with torch.no_grad():
            pred_actions = self.policy.predict_action(batched_obs)
            if pred_actions.dim() == 3 and pred_actions.shape[1] == 1:
                pred_actions = pred_actions.squeeze(1)

        diff = actions_batch - pred_actions.detach()
        logprobs = -0.5 * (diff ** 2).sum(dim=-1)
        return logprobs

    def train_on_rollouts(self, rollout: RolloutData) -> dict[str, float]:
        """Run GRPO training on collected rollouts.

        Args:
            rollout: collected rollout data

        Returns:
            dict with averaged metrics over update epochs
        """
        self.policy.train()

        # Scale rewards
        scaled_rewards = rollout.episode_returns * self.cfg.grpo_cfg.reward_coef

        # Compute group-relative advantages
        advantages = self.grpo.compute_advantages(scaled_rewards)

        # Multi-epoch updates
        all_metrics = []
        for epoch in range(self.cfg.grpo_cfg.update_epochs):
            # Re-evaluate logprobs under current policy
            new_logprobs = self._recompute_logprobs(
                rollout.obs,
                rollout.actions,
            )

            # GRPO update
            metrics = self.grpo.update(
                logprobs=new_logprobs,
                old_logprobs=rollout.logprobs,
                advantages=advantages,
                optimizer=self.optimizer,
            )
            all_metrics.append(metrics)

        # Average metrics across epochs
        avg_metrics = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics if key in m]
            avg_metrics[key] = sum(vals) / len(vals)

        return avg_metrics

    def learn(self, num_iterations: int = 100):
        """Main training loop.

        Each iteration:
        1. Collect group_size rollouts per env
        2. Compute GRPO advantages
        3. Update policy for update_epochs
        """
        print(f"GRPO Training: {num_iterations} iterations, "
              f"group_size={self.cfg.grpo_cfg.group_size}, "
              f"num_envs={self.env.num_envs}, "
              f"device={self.device}")

        for iteration in range(num_iterations):
            t_start = time.time()

            # 1. Collect rollouts
            with torch.no_grad():
                rollout = self.collect_rollouts()

            # 2. Train
            metrics = self.train_on_rollouts(rollout)

            self.global_step += 1
            elapsed = time.time() - t_start

            # Logging
            mean_return = rollout.episode_returns.mean().item()
            success_rate = rollout.dones.float().mean().item()

            self.writer.add_scalar("train/mean_return", mean_return, self.global_step)
            self.writer.add_scalar("train/success_rate", success_rate, self.global_step)
            self.writer.add_scalar("train/total_loss", metrics.get("total_loss", 0), self.global_step)
            self.writer.add_scalar("train/policy_loss", metrics.get("policy_loss", 0), self.global_step)
            if "grpo/approx_kl" in metrics:
                self.writer.add_scalar("train/approx_kl", metrics["grpo/approx_kl"], self.global_step)
            if "grpo/clip_fraction" in metrics:
                self.writer.add_scalar("train/clip_fraction", metrics["grpo/clip_fraction"], self.global_step)

            if iteration % self.cfg.log_interval == 0:
                print(f"Iter {iteration+1}/{num_iterations} | "
                      f"return={mean_return:.3f} | "
                      f"success={success_rate:.2%} | "
                      f"loss={metrics.get('total_loss', 0):.4f} | "
                      f"kl={metrics.get('grpo/approx_kl', 0):.4f} | "
                      f"time={elapsed:.1f}s")

            if self.cfg.save_interval > 0 and (iteration + 1) % self.cfg.save_interval == 0:
                self.save_checkpoint(tag=f"iter_{iteration+1}")

        self.save_checkpoint(tag="final")
        print("GRPO training complete.")

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 50) -> dict[str, float]:
        """Run deterministic evaluation episodes.

        Returns:
            dict with mean_return, success_rate
        """
        self.policy.eval()
        num_envs = self.env.num_envs
        num_batches = (num_episodes + num_envs - 1) // num_envs

        all_returns = []
        all_successes = []

        for _ in range(num_batches):
            obs, _ = self.env.reset()
            episode_return = torch.zeros(num_envs, device=self.device)
            done_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

            for t in range(self.env.max_episode_length):
                actions = self.policy.predict_action(obs)
                if actions.dim() == 3 and actions.shape[1] == 1:
                    actions = actions.squeeze(1)

                obs, rewards, dones, extras = self.env.step(actions)
                episode_return += rewards * (~done_mask).float()
                done_mask = done_mask | dones

                if done_mask.all():
                    break

            all_returns.append(episode_return)
            all_successes.append(done_mask.float())

        returns = torch.cat(all_returns)[:num_episodes]
        successes = torch.cat(all_successes)[:num_episodes]

        return {
            "mean_return": returns.mean().item(),
            "success_rate": successes.mean().item(),
        }

    def save_checkpoint(self, tag: str = "latest"):
        path = self.checkpoint_dir / f"grpo_checkpoint_{tag}.pt"
        torch.save({
            "global_step": self.global_step,
            "total_episodes": self.total_episodes,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.total_episodes = ckpt.get("total_episodes", 0)
        print(f"Checkpoint loaded: step={self.global_step}, episodes={self.total_episodes}")


def _batch_obs_list(obs_list: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    """Batch a list of single-sample obs dicts into one batched dict."""
    if not obs_list:
        return {}

    keys = obs_list[0].keys()
    batched = {}
    for k in keys:
        vals = [obs[k] for obs in obs_list]
        if isinstance(vals[0], torch.Tensor):
            batched[k] = torch.cat(vals, dim=0).to(device)
        elif isinstance(vals[0], list):
            batched[k] = [item for sublist in vals for item in sublist]
        else:
            batched[k] = vals
    return batched


class _DummyWriter:
    def add_scalar(self, *args, **kwargs): pass
    def close(self): pass


@configclass
class VLAGRPORunnerCfg(ModuleBaseCfg):
    """VLA GRPO runner configuration."""

    class_type: type[VLAGRPORunner] = VLAGRPORunner

    grpo_cfg: GRPOAlgorithmCfg = GRPOAlgorithmCfg()

    # Logging
    log_interval: int = 1
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints/grpo"

    # Evaluation
    eval_interval: int = 10
    eval_episodes: int = 50
