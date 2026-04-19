"""
Tests for GRPO algorithm and VLA GRPO Runner.

Covers:
1. GRPO advantage computation (group-relative normalization)
2. Clipped surrogate loss
3. GRPOAlgorithm update
4. End-to-end training with mock env + MLP policy
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Any
from unittest.mock import MagicMock


# ---- GRPO Advantage Tests ----

class TestGRPOAdvantages:
    def test_basic_shape(self):
        from RoboRenForce.algorithms.vla_training.grpo import compute_grpo_advantages
        rewards = torch.tensor([1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0])
        adv = compute_grpo_advantages(rewards, group_size=4)
        assert adv.shape == rewards.shape

    def test_zero_mean_within_group(self):
        from RoboRenForce.algorithms.vla_training.grpo import compute_grpo_advantages
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0])
        adv = compute_grpo_advantages(rewards, group_size=4)
        # Each group should have ~zero mean after normalization
        group1 = adv[:4]
        group2 = adv[4:]
        assert abs(group1.mean().item()) < 1e-5
        assert abs(group2.mean().item()) < 1e-5

    def test_unit_std_within_group(self):
        from RoboRenForce.algorithms.vla_training.grpo import compute_grpo_advantages
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = compute_grpo_advantages(rewards, group_size=4)
        # Should be z-normalized: std ≈ 1
        assert abs(adv.std().item() - 1.0) < 0.1

    def test_constant_rewards_zero_advantage(self):
        from RoboRenForce.algorithms.vla_training.grpo import compute_grpo_advantages
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        adv = compute_grpo_advantages(rewards, group_size=4)
        # All same reward → all advantages should be ~0
        assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-5)

    def test_two_groups_independent(self):
        from RoboRenForce.algorithms.vla_training.grpo import compute_grpo_advantages
        # Group 1: [0, 0, 0, 1], Group 2: [10, 10, 10, 11]
        rewards = torch.tensor([0.0, 0.0, 0.0, 1.0,  10.0, 10.0, 10.0, 11.0])
        adv = compute_grpo_advantages(rewards, group_size=4)
        # Both groups should produce the same advantage pattern
        assert torch.allclose(adv[:4], adv[4:], atol=1e-5)

    def test_higher_reward_gets_positive_advantage(self):
        from RoboRenForce.algorithms.vla_training.grpo import compute_grpo_advantages
        rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
        adv = compute_grpo_advantages(rewards, group_size=4)
        # The one with reward=1 should have the highest advantage
        assert adv[3] > adv[0]
        assert adv[3] > 0


# ---- Clipped Surrogate Loss Tests ----

class TestClippedSurrogateLoss:
    def test_basic_loss_computation(self):
        from RoboRenForce.algorithms.vla_training.grpo import compute_clipped_surrogate_loss
        logprobs = torch.tensor([-1.0, -2.0, -1.5, -0.5])
        old_logprobs = torch.tensor([-1.0, -2.0, -1.5, -0.5])
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

        loss, metrics = compute_clipped_surrogate_loss(logprobs, old_logprobs, advantages)
        # When logprobs == old_logprobs, ratio = 1, loss = -mean(advantages) for positive adv
        assert isinstance(loss.item(), float)
        assert "clip_fraction" in metrics
        assert "approx_kl" in metrics

    def test_no_clip_when_ratio_is_one(self):
        from RoboRenForce.algorithms.vla_training.grpo import compute_clipped_surrogate_loss
        logprobs = torch.tensor([-1.0, -2.0])
        old_logprobs = logprobs.clone()
        advantages = torch.tensor([1.0, 1.0])

        loss, metrics = compute_clipped_surrogate_loss(logprobs, old_logprobs, advantages)
        assert metrics["clip_fraction"].item() == 0.0

    def test_gradient_flows(self):
        from RoboRenForce.algorithms.vla_training.grpo import compute_clipped_surrogate_loss
        logprobs = torch.tensor([-1.0, -2.0], requires_grad=True)
        old_logprobs = torch.tensor([-1.1, -1.9])
        advantages = torch.tensor([1.0, -0.5])

        loss, _ = compute_clipped_surrogate_loss(logprobs, old_logprobs, advantages)
        loss.backward()
        assert logprobs.grad is not None

    def test_large_ratio_gets_clipped(self):
        from RoboRenForce.algorithms.vla_training.grpo import compute_clipped_surrogate_loss
        # Large ratio = big policy change → should be clipped
        logprobs = torch.tensor([0.0])       # exp(0 - (-5)) = exp(5) ≈ 148
        old_logprobs = torch.tensor([-5.0])
        advantages = torch.tensor([1.0])

        _, metrics = compute_clipped_surrogate_loss(
            logprobs, old_logprobs, advantages, clip_ratio=0.2)
        assert metrics["clip_fraction"].item() == 1.0


# ---- GRPOAlgorithm Tests ----

class TestGRPOAlgorithm:
    def test_compute_advantages(self):
        from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithm, GRPOAlgorithmCfg
        cfg = GRPOAlgorithmCfg(group_size=4)
        grpo = GRPOAlgorithm(cfg)

        rewards = torch.tensor([0.0, 0.5, 0.5, 1.0])
        adv = grpo.compute_advantages(rewards)
        assert adv.shape == (4,)
        assert adv[3] > adv[0]  # highest reward → highest advantage

    def test_compute_loss(self):
        from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithm, GRPOAlgorithmCfg
        cfg = GRPOAlgorithmCfg(group_size=4, kl_beta=0, entropy_bonus=0)
        grpo = GRPOAlgorithm(cfg)

        logprobs = torch.tensor([-1.0, -2.0, -1.5, -0.5])
        old_logprobs = logprobs.clone()
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

        loss_dict = grpo.compute_loss(logprobs, old_logprobs, advantages)
        assert "total_loss" in loss_dict
        assert "policy_loss" in loss_dict

    def test_update_step(self):
        from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithm, GRPOAlgorithmCfg
        cfg = GRPOAlgorithmCfg(group_size=4, kl_beta=0, entropy_bonus=0)
        grpo = GRPOAlgorithm(cfg)

        # Create a simple model to have optimizer params
        model = nn.Linear(4, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        logprobs = torch.tensor([-1.0, -2.0, -1.5, -0.5], requires_grad=True)
        old_logprobs = torch.tensor([-1.0, -2.0, -1.5, -0.5])
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

        # Override optimizer params to include logprobs
        optimizer = torch.optim.SGD([logprobs], lr=1e-3)
        metrics = grpo.update(logprobs, old_logprobs, advantages, optimizer)

        assert "total_loss" in metrics
        assert isinstance(metrics["total_loss"], float)

    def test_kl_penalty(self):
        from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithm, GRPOAlgorithmCfg
        cfg = GRPOAlgorithmCfg(group_size=4, kl_beta=0.1, entropy_bonus=0)
        grpo = GRPOAlgorithm(cfg)

        logprobs = torch.tensor([-1.0, -2.0, -1.5, -0.5])
        old_logprobs = logprobs.clone()
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        ref_logprobs = torch.tensor([-1.5, -2.5, -1.0, -1.0])

        loss_dict = grpo.compute_loss(logprobs, old_logprobs, advantages,
                                       ref_logprobs=ref_logprobs)
        assert "kl_loss" in loss_dict
        assert "kl_div" in loss_dict


# ---- End-to-End Training Test ----

class MockEmbodiedEnv:
    """Mock environment for testing the GRPO runner without a real simulator."""

    def __init__(self, num_envs=4, state_dim=8, action_dim=4, max_steps=10):
        self.num_envs = num_envs
        self.num_actions = action_dim
        self.state_dim = state_dim
        self.max_episode_length = max_steps
        self.device = torch.device("cpu")
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        obs = {
            "states": torch.randn(self.num_envs, self.state_dim),
            "task_descriptions": ["test task"] * self.num_envs,
        }
        return obs, {}

    def step(self, actions):
        self._step_count += 1
        obs = {
            "states": torch.randn(self.num_envs, self.state_dim),
            "task_descriptions": ["test task"] * self.num_envs,
        }
        # Random binary rewards
        rewards = (torch.rand(self.num_envs) > 0.7).float()
        # Done after max_steps
        dones = torch.full((self.num_envs,), self._step_count >= self.max_episode_length)
        extras = {"termination": rewards > 0, "timeout": dones & (rewards == 0)}
        return obs, rewards, dones, extras


class TestGRPORunnerE2E:
    def test_runner_init(self):
        from RoboRenForce.runners.vla.rl import VLAGRPORunner, VLAGRPORunnerCfg
        from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithmCfg
        from RRF_models.mlp_baseline.mlp_policy import MLPBaselinePolicy

        env = MockEmbodiedEnv(num_envs=4, state_dim=8, action_dim=4)
        policy = MLPBaselinePolicy(state_dim=8, action_dim=4)
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(group_size=2, update_epochs=1),
            checkpoint_dir="/tmp/test_grpo_ckpt",
        )
        runner = VLAGRPORunner(cfg, env, policy, device="cpu",
                               log_dir="/tmp/test_grpo_logs")
        assert runner.global_step == 0

    def test_collect_rollouts(self):
        from RoboRenForce.runners.vla.rl import VLAGRPORunner, VLAGRPORunnerCfg
        from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithmCfg
        from RRF_models.mlp_baseline.mlp_policy import MLPBaselinePolicy

        env = MockEmbodiedEnv(num_envs=4, state_dim=8, action_dim=4, max_steps=5)
        policy = MLPBaselinePolicy(state_dim=8, action_dim=4)
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(group_size=3, update_epochs=1),
            checkpoint_dir="/tmp/test_grpo_ckpt",
        )
        runner = VLAGRPORunner(cfg, env, policy, device="cpu",
                               log_dir="/tmp/test_grpo_logs")

        rollout = runner.collect_rollouts()
        total = env.num_envs * 3  # 4 envs × 3 group_size = 12
        assert rollout.actions.shape == (total, 4)
        assert rollout.logprobs.shape == (total,)
        assert rollout.rewards.shape == (total,)
        assert rollout.episode_returns.shape == (total,)
        assert len(rollout.obs) == total

    def test_train_on_rollouts(self):
        from RoboRenForce.runners.vla.rl import VLAGRPORunner, VLAGRPORunnerCfg
        from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithmCfg
        from RRF_models.mlp_baseline.mlp_policy import MLPBaselinePolicy

        env = MockEmbodiedEnv(num_envs=4, state_dim=8, action_dim=4, max_steps=5)
        policy = MLPBaselinePolicy(state_dim=8, action_dim=4)
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(
                group_size=2, update_epochs=2, kl_beta=0, entropy_bonus=0,
                learning_rate=1e-3,
            ),
            checkpoint_dir="/tmp/test_grpo_ckpt",
        )
        runner = VLAGRPORunner(cfg, env, policy, device="cpu",
                               log_dir="/tmp/test_grpo_logs")

        rollout = runner.collect_rollouts()
        metrics = runner.train_on_rollouts(rollout)

        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert isinstance(metrics["total_loss"], float)

    def test_learn_loop(self):
        """Full training loop: 3 iterations, should not crash."""
        from RoboRenForce.runners.vla.rl import VLAGRPORunner, VLAGRPORunnerCfg
        from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithmCfg
        from RRF_models.mlp_baseline.mlp_policy import MLPBaselinePolicy

        env = MockEmbodiedEnv(num_envs=2, state_dim=8, action_dim=4, max_steps=5)
        policy = MLPBaselinePolicy(state_dim=8, action_dim=4)
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(
                group_size=2, update_epochs=1, kl_beta=0, entropy_bonus=0,
                learning_rate=1e-3,
            ),
            save_interval=0,  # disable checkpoint saving in test
            checkpoint_dir="/tmp/test_grpo_ckpt",
        )
        runner = VLAGRPORunner(cfg, env, policy, device="cpu",
                               log_dir="/tmp/test_grpo_logs")

        # Should complete without error
        runner.learn(num_iterations=3)
        assert runner.global_step == 3
        assert runner.total_episodes == 2 * 2 * 3  # 2 envs × 2 group × 3 iters = 12

    def test_evaluate(self):
        from RoboRenForce.runners.vla.rl import VLAGRPORunner, VLAGRPORunnerCfg
        from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithmCfg
        from RRF_models.mlp_baseline.mlp_policy import MLPBaselinePolicy

        env = MockEmbodiedEnv(num_envs=4, state_dim=8, action_dim=4, max_steps=5)
        policy = MLPBaselinePolicy(state_dim=8, action_dim=4)
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(group_size=2),
            checkpoint_dir="/tmp/test_grpo_ckpt",
        )
        runner = VLAGRPORunner(cfg, env, policy, device="cpu",
                               log_dir="/tmp/test_grpo_logs")

        eval_metrics = runner.evaluate(num_episodes=8)
        assert "mean_return" in eval_metrics
        assert "success_rate" in eval_metrics
        assert 0 <= eval_metrics["success_rate"] <= 1

    def test_checkpoint_roundtrip(self):
        import tempfile
        from RoboRenForce.runners.vla.rl import VLAGRPORunner, VLAGRPORunnerCfg
        from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithmCfg
        from RRF_models.mlp_baseline.mlp_policy import MLPBaselinePolicy

        env = MockEmbodiedEnv(num_envs=2, state_dim=8, action_dim=4, max_steps=5)
        policy = MLPBaselinePolicy(state_dim=8, action_dim=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = VLAGRPORunnerCfg(
                grpo_cfg=GRPOAlgorithmCfg(group_size=2, update_epochs=1,
                                           kl_beta=0, entropy_bonus=0),
                checkpoint_dir=tmpdir,
            )
            runner = VLAGRPORunner(cfg, env, policy, device="cpu",
                                   log_dir=f"{tmpdir}/logs")

            # Train a bit
            runner.learn(num_iterations=2)
            runner.save_checkpoint(tag="test")

            # Load into fresh runner
            policy2 = MLPBaselinePolicy(state_dim=8, action_dim=4)
            runner2 = VLAGRPORunner(cfg, env, policy2, device="cpu",
                                    log_dir=f"{tmpdir}/logs2")
            runner2.load_checkpoint(f"{tmpdir}/grpo_checkpoint_test.pt")

            assert runner2.global_step == 2
            # Check weights match
            for p1, p2 in zip(policy.parameters(), policy2.parameters()):
                assert torch.allclose(p1, p2)
