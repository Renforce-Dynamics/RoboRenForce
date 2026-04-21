"""
VLA PPO Runner — RL fine-tuning with actor-critic PPO.

Unlike GRPO (group-relative advantages, no value function), PPO uses a
learned value function and GAE for per-step advantage estimation.

Training loop:
    for each iteration:
        1. Collect rollouts: obs, actions, logprobs, values, rewards, dones
        2. Compute GAE advantages and returns
        3. For update_epochs:
             a. Re-evaluate logprobs and values under current policy
             b. Actor loss: clipped surrogate with GAE advantages
             c. Critic loss: clipped value function loss (Huber)
             d. Update policy

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
from RoboRenForce.algorithms.vla_training.ppo import PPOAlgorithm, PPOAlgorithmCfg


@dataclass
class PPORolloutData:
    """Collected rollout data with per-step values for GAE."""
    obs: list[dict[str, Any]]       # [T * num_envs] obs dicts
    actions: torch.Tensor           # [T * num_envs, action_dim]
    logprobs: torch.Tensor          # [T * num_envs]
    values: torch.Tensor            # [T+1, num_envs] (includes bootstrap)
    rewards: torch.Tensor           # [T, num_envs]
    dones: torch.Tensor             # [T, num_envs]


class VLAPPORunner(ModuleBase):
    """VLA RL fine-tuning runner using PPO with GAE.

    Usage:
        runner = VLAPPORunner(cfg, env, policy, device="cuda:0")
        runner.learn(num_iterations=100)
    """

    def __init__(
        self,
        cfg: "VLAPPORunnerCfg",
        env: EmbodiedEnv,
        policy: BasePolicy,
        device: str = "cpu",
        log_dir: str = "logs/ppo",
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

        self.ppo = PPOAlgorithm(cfg.ppo_cfg)

        self.optimizer = torch.optim.AdamW(
            list(policy.trainable_parameters()),
            lr=cfg.ppo_cfg.learning_rate,
            weight_decay=cfg.ppo_cfg.weight_decay,
        )

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
    def collect_rollouts(self) -> PPORolloutData:
        """Collect one full rollout episode from all envs.

        Records per-step obs, actions, logprobs, values, rewards, dones
        for GAE computation.
        """
        self.policy.eval()
        num_envs = self.env.num_envs
        max_steps = self.env.max_episode_length

        all_obs = []
        all_actions = []
        all_logprobs = []
        all_values = []
        all_rewards = []
        all_dones = []

        obs, info = self.env.reset()

        for t in range(max_steps):
            # Get action, logprob, value from policy
            result = self._get_action_value(obs)
            actions = result["actions"]
            logprobs = result["logprobs"]
            values = result["values"]

            all_obs.append(obs)
            all_actions.append(actions)
            all_logprobs.append(logprobs)
            all_values.append(values)

            # Step env
            next_obs, rewards, dones, extras = self.env.step(actions)
            all_rewards.append(rewards)
            all_dones.append(dones)

            obs = next_obs

            if dones.all():
                break

        # Bootstrap value for last state
        bootstrap_result = self._get_action_value(obs)
        all_values.append(bootstrap_result["values"])

        T = len(all_rewards)
        self.total_episodes += num_envs

        # Stack tensors: [T, num_envs, ...]
        stacked_actions = torch.stack(all_actions, dim=0)  # [T, B, action_dim]
        stacked_logprobs = torch.stack(all_logprobs, dim=0)  # [T, B]
        stacked_values = torch.stack(all_values, dim=0)  # [T+1, B]
        stacked_rewards = torch.stack(all_rewards, dim=0)  # [T, B]
        stacked_dones = torch.stack(all_dones, dim=0)  # [T, B]

        # Flatten obs for training: [T * B] list
        flat_obs = []
        for t_obs in all_obs:
            for env_idx in range(num_envs):
                flat_obs.append(
                    {k: v[env_idx:env_idx+1] if isinstance(v, torch.Tensor)
                     else [v[env_idx]] if isinstance(v, list) else v
                     for k, v in t_obs.items()}
                )

        return PPORolloutData(
            obs=flat_obs,
            actions=stacked_actions.reshape(-1, stacked_actions.shape[-1]),
            logprobs=stacked_logprobs.reshape(-1),
            values=stacked_values,
            rewards=stacked_rewards,
            dones=stacked_dones,
        )

    @torch.no_grad()
    def _get_action_value(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Get actions, logprobs, and values from policy."""
        try:
            result = self.policy.forward(ForwardType.PPO, obs=obs, actions=None)
            actions = result.get("pred_actions", result.get("actions"))
            if actions is not None and actions.dim() == 3 and actions.shape[1] == 1:
                actions = actions.squeeze(1)

            values = result.get("values")
            if values is None:
                values = self.policy.get_value(obs)
            if values is None:
                values = torch.zeros(actions.shape[0], device=actions.device)

            logprobs = result.get("logprobs")
            if logprobs is None:
                logprobs = -0.5 * (actions ** 2).sum(dim=-1)

            return {"actions": actions, "logprobs": logprobs, "values": values}
        except (NotImplementedError, TypeError):
            pass

        # Fallback
        actions = self.policy.predict_action(obs)
        if actions.dim() == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)
        logprobs = -0.5 * (actions ** 2).sum(dim=-1)

        values = self.policy.get_value(obs)
        if values is None:
            values = torch.zeros(actions.shape[0], device=actions.device)

        return {"actions": actions, "logprobs": logprobs, "values": values}

    def _recompute_logprobs_values(
        self,
        obs_batch: list[dict[str, Any]],
        actions_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate logprobs and values under current policy."""
        from RoboRenForce.runners.vla.rl.vla_grpo_runner import _batch_obs_list
        batched_obs = _batch_obs_list(obs_batch, self.device)

        try:
            result = self.policy.forward(
                ForwardType.PPO, obs=batched_obs, actions=actions_batch,
            )
            logprobs = result.get("logprobs")
            values = result.get("values")
            if logprobs is None:
                with torch.no_grad():
                    pred = self.policy.predict_action(batched_obs)
                    if pred.dim() == 3 and pred.shape[1] == 1:
                        pred = pred.squeeze(1)
                diff = actions_batch - pred.detach()
                logprobs = -0.5 * (diff ** 2).sum(dim=-1)
            if values is None:
                values = torch.zeros(actions_batch.shape[0], device=self.device)
            return logprobs, values
        except (NotImplementedError, TypeError):
            pass

        with torch.no_grad():
            pred = self.policy.predict_action(batched_obs)
            if pred.dim() == 3 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
        diff = actions_batch - pred.detach()
        logprobs = -0.5 * (diff ** 2).sum(dim=-1)
        values = torch.zeros(actions_batch.shape[0], device=self.device)
        return logprobs, values

    def train_on_rollouts(self, rollout: PPORolloutData) -> dict[str, float]:
        """Run PPO training on collected rollouts."""
        self.policy.train()
        num_envs = self.env.num_envs

        # Scale rewards
        scaled_rewards = rollout.rewards * self.cfg.ppo_cfg.reward_coef

        # Compute GAE
        advantages, returns = self.ppo.compute_advantages(
            scaled_rewards, rollout.values, rollout.dones,
        )

        # Flatten for training
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_old_values = rollout.values[:-1].reshape(-1)

        all_metrics = []
        for epoch in range(self.cfg.ppo_cfg.update_epochs):
            new_logprobs, new_values = self._recompute_logprobs_values(
                rollout.obs, rollout.actions,
            )
            metrics = self.ppo.update(
                logprobs=new_logprobs,
                old_logprobs=rollout.logprobs,
                advantages=flat_advantages,
                values=new_values,
                returns=flat_returns,
                old_values=flat_old_values,
                optimizer=self.optimizer,
            )
            all_metrics.append(metrics)

        avg_metrics = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics if key in m]
            avg_metrics[key] = sum(vals) / len(vals)
        return avg_metrics

    def learn(self, num_iterations: int = 100):
        """Main training loop."""
        print(f"PPO Training: {num_iterations} iterations, "
              f"num_envs={self.env.num_envs}, device={self.device}")

        for iteration in range(num_iterations):
            t_start = time.time()

            with torch.no_grad():
                rollout = self.collect_rollouts()

            metrics = self.train_on_rollouts(rollout)
            self.global_step += 1
            elapsed = time.time() - t_start

            mean_return = rollout.rewards.sum(dim=0).mean().item()

            self.writer.add_scalar("train/mean_return", mean_return, self.global_step)
            self.writer.add_scalar("train/total_loss", metrics.get("total_loss", 0), self.global_step)
            self.writer.add_scalar("train/policy_loss", metrics.get("policy_loss", 0), self.global_step)
            self.writer.add_scalar("train/value_loss", metrics.get("value_loss", 0), self.global_step)

            if iteration % self.cfg.log_interval == 0:
                print(f"Iter {iteration+1}/{num_iterations} | "
                      f"return={mean_return:.3f} | "
                      f"loss={metrics.get('total_loss', 0):.4f} | "
                      f"v_loss={metrics.get('value_loss', 0):.4f} | "
                      f"time={elapsed:.1f}s")

            if self.cfg.save_interval > 0 and (iteration + 1) % self.cfg.save_interval == 0:
                self.save_checkpoint(tag=f"iter_{iteration+1}")

        self.save_checkpoint(tag="final")
        print("PPO training complete.")

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 50) -> dict[str, float]:
        """Run deterministic evaluation."""
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
        path = self.checkpoint_dir / f"ppo_checkpoint_{tag}.pt"
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


class _DummyWriter:
    def add_scalar(self, *args, **kwargs): pass
    def close(self): pass


@configclass
class VLAPPORunnerCfg(ModuleBaseCfg):
    """VLA PPO runner configuration."""

    class_type: type[VLAPPORunner] = VLAPPORunner

    ppo_cfg: PPOAlgorithmCfg = PPOAlgorithmCfg()

    log_interval: int = 1
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints/ppo"

    eval_interval: int = 10
    eval_episodes: int = 50
