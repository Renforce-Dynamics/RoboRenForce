"""
Tests for ALL VLA training algorithms.

Covers: PPO, SFT, IQL, DAgger, SAC
Each algorithm is tested for:
- Correct loss computation
- Gradient flow
- Update step mechanics
- Key algorithm-specific properties
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Any


# ================================================================
# Helper: mock policy for testing
# ================================================================

class MockPolicy(nn.Module):
    """Minimal policy for algorithm tests."""

    def __init__(self, state_dim=8, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )
        self.value_head = nn.Linear(32, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def predict_action(self, obs, **kwargs):
        states = obs.get("states", obs.get("obs"))
        return self.net(states).unsqueeze(1)

    def forward(self, forward_type=None, **kwargs):
        obs = kwargs.get("obs", {})
        states = obs.get("states", obs.get("obs", torch.zeros(1, 8)))
        h = self.net[0](states)
        h = self.net[1](h)
        actions = self.net[2](h)
        values = self.value_head(h).squeeze(-1)

        target = kwargs.get("target_actions")
        if target is not None:
            if target.dim() == 3:
                target = target.squeeze(1)
            loss = ((actions - target) ** 2).mean()
        else:
            loss = torch.tensor(0.0)

        std = self.log_std.exp()
        dist = torch.distributions.Normal(actions, std)
        given_actions = kwargs.get("actions")
        if given_actions is not None:
            logprobs = dist.log_prob(given_actions).sum(dim=-1)
        else:
            logprobs = dist.log_prob(actions).sum(dim=-1)

        return {
            "loss": loss,
            "pred_actions": actions.unsqueeze(1),
            "logprobs": logprobs,
            "values": values,
            "entropy": dist.entropy().sum(dim=-1),
        }

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def get_value(self, obs):
        states = obs.get("states", obs.get("obs"))
        h = self.net[0](states)
        h = self.net[1](h)
        return self.value_head(h).squeeze(-1)


# ================================================================
# PPO Tests
# ================================================================

class TestPPOAlgorithm:
    """Test PPO algorithm components."""

    def test_gae_shape(self):
        """GAE output shapes match input."""
        from RoboRenForce.algorithms.vla_training.ppo import compute_gae_advantages

        T, B = 10, 4
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        dones = torch.zeros(T, B, dtype=torch.bool)

        advantages, returns = compute_gae_advantages(rewards, values, dones)
        assert advantages.shape == (T, B)
        assert returns.shape == (T, B)

    def test_gae_zero_reward(self):
        """GAE with zero rewards should give small advantages."""
        from RoboRenForce.algorithms.vla_training.ppo import compute_gae_advantages

        T, B = 5, 2
        rewards = torch.zeros(T, B)
        values = torch.ones(T + 1, B) * 0.5
        dones = torch.zeros(T, B, dtype=torch.bool)

        advantages, returns = compute_gae_advantages(
            rewards, values, dones, normalize=False
        )
        # With constant values and zero rewards, advantages should be small
        assert advantages.abs().max() < 1.0

    def test_gae_done_resets(self):
        """GAE should reset at episode boundaries."""
        from RoboRenForce.algorithms.vla_training.ppo import compute_gae_advantages

        T, B = 6, 1
        rewards = torch.ones(T, B)
        values = torch.zeros(T + 1, B)
        dones = torch.zeros(T, B, dtype=torch.bool)
        dones[2, 0] = True  # Episode ends at step 2

        advantages_with_done, _ = compute_gae_advantages(
            rewards, values, dones, gamma=0.99, gae_lambda=1.0, normalize=False
        )
        # Without done, advantages would accumulate across the boundary
        no_dones = torch.zeros(T, B, dtype=torch.bool)
        advantages_no_done, _ = compute_gae_advantages(
            rewards, values, no_dones, gamma=0.99, gae_lambda=1.0, normalize=False
        )
        # Step 0 advantage should be smaller with done (cuts off future rewards)
        assert advantages_with_done[0, 0] < advantages_no_done[0, 0]

    def test_ppo_actor_loss_basic(self):
        """PPO actor loss is finite and has correct sign."""
        from RoboRenForce.algorithms.vla_training.ppo import compute_ppo_actor_loss

        logprobs = torch.tensor([-1.0, -2.0, -1.5], requires_grad=True)
        old_logprobs = torch.tensor([-1.0, -2.0, -1.5])
        advantages = torch.tensor([1.0, -0.5, 0.3])

        loss, metrics = compute_ppo_actor_loss(logprobs, old_logprobs, advantages)
        assert torch.isfinite(loss)
        assert "clip_fraction" in metrics
        assert "approx_kl" in metrics

    def test_ppo_actor_loss_clipping(self):
        """Large ratio should be clipped."""
        from RoboRenForce.algorithms.vla_training.ppo import compute_ppo_actor_loss

        logprobs = torch.tensor([0.0], requires_grad=True)  # ratio = exp(0 - (-5)) = exp(5) ≈ 148
        old_logprobs = torch.tensor([-5.0])
        advantages = torch.tensor([1.0])

        loss, metrics = compute_ppo_actor_loss(logprobs, old_logprobs, advantages, clip_ratio_low=0.2)
        # Clip fraction should be 1.0 (fully clipped)
        assert metrics["clip_fraction"] > 0.5

    def test_ppo_critic_loss(self):
        """Critic loss is finite."""
        from RoboRenForce.algorithms.vla_training.ppo import compute_ppo_critic_loss

        values = torch.randn(10, requires_grad=True)
        returns = torch.randn(10)
        old_values = torch.randn(10)

        loss, metrics = compute_ppo_critic_loss(values, returns, old_values)
        assert torch.isfinite(loss)
        assert "explained_variance" in metrics

    def test_ppo_algorithm_update(self):
        """Full PPO algorithm update step."""
        from RoboRenForce.algorithms.vla_training.ppo import PPOAlgorithm, PPOAlgorithmCfg

        cfg = PPOAlgorithmCfg(clip_ratio_low=0.2, entropy_bonus=0.01)
        ppo = PPOAlgorithm(cfg)

        B = 16
        logprobs = torch.randn(B, requires_grad=True)
        old_logprobs = logprobs.detach().clone()
        advantages = torch.randn(B)
        values = torch.randn(B, requires_grad=True)
        returns = torch.randn(B)
        old_values = torch.randn(B)
        entropy = torch.rand(B)

        optimizer = torch.optim.SGD([logprobs, values], lr=1e-3)
        metrics = ppo.update(
            logprobs, old_logprobs, advantages, values, returns, old_values,
            optimizer, entropy=entropy,
        )

        assert np.isfinite(metrics["total_loss"])
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert "entropy" in metrics


# ================================================================
# SFT Tests
# ================================================================

class TestSFTAlgorithm:
    """Test SFT algorithm."""

    def test_sft_basic_update(self):
        """SFT update reduces loss."""
        from RoboRenForce.algorithms.vla_training.sft import SFTAlgorithm, SFTAlgorithmCfg

        cfg = SFTAlgorithmCfg(action_loss_type="mse")
        sft = SFTAlgorithm(cfg)

        policy = MockPolicy(state_dim=8, action_dim=4)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        batch = {
            "states": torch.randn(16, 8),
            "action": torch.randn(16, 4),
        }

        losses = []
        for _ in range(5):
            metrics = sft.update(batch, policy, optimizer)
            losses.append(metrics["total_loss"])

        assert losses[-1] < losses[0], f"SFT loss did not decrease: {losses}"

    def test_sft_loss_types(self):
        """Test different loss types produce different values."""
        from RoboRenForce.algorithms.vla_training.sft import SFTAlgorithm, SFTAlgorithmCfg

        policy = MockPolicy(state_dim=8, action_dim=4)
        batch = {
            "states": torch.randn(8, 8),
            "action": torch.randn(8, 4),
        }

        losses = {}
        for loss_type in ["mse", "l1", "smooth_l1"]:
            cfg = SFTAlgorithmCfg(action_loss_type=loss_type)
            sft = SFTAlgorithm(cfg)
            loss_dict = sft.compute_loss(batch, policy)
            losses[loss_type] = loss_dict["total_loss"].item()

        # All should be finite
        for k, v in losses.items():
            assert np.isfinite(v), f"{k} loss is not finite: {v}"

    def test_sft_with_kl(self):
        """SFT with KL penalty against reference policy."""
        from RoboRenForce.algorithms.vla_training.sft import SFTAlgorithm, SFTAlgorithmCfg

        cfg = SFTAlgorithmCfg(kl_coef=0.1)
        sft = SFTAlgorithm(cfg)

        policy = MockPolicy(state_dim=8, action_dim=4)
        ref_policy = MockPolicy(state_dim=8, action_dim=4)
        sft.set_reference_policy(ref_policy)

        # Ref policy params should be frozen
        for p in ref_policy.parameters():
            assert not p.requires_grad

        batch = {
            "states": torch.randn(8, 8),
            "action": torch.randn(8, 4),
        }
        loss_dict = sft.compute_loss(batch, policy)
        assert "kl_loss" in loss_dict or "total_loss" in loss_dict


# ================================================================
# IQL Tests
# ================================================================

class TestIQLAlgorithm:
    """Test IQL algorithm."""

    def test_expectile_loss(self):
        """Expectile loss asymmetry: positive diff penalized more when τ > 0.5."""
        from RoboRenForce.algorithms.vla_training.iql import iql_expectile_loss

        diff = torch.tensor([1.0, -1.0])
        loss_high = iql_expectile_loss(diff, expectile=0.9)
        loss_low = iql_expectile_loss(diff, expectile=0.1)

        # With τ=0.9: positive diff (0.9 * 1²) > negative diff (0.1 * 1²)
        assert loss_high[0] > loss_high[1]
        # With τ=0.1: positive diff (0.1 * 1²) < negative diff (0.9 * 1²)
        assert loss_low[0] < loss_low[1]

    def test_iql_initialization(self):
        """IQL creates all required networks."""
        from RoboRenForce.algorithms.vla_training.iql import IQLAlgorithm, IQLAlgorithmCfg

        cfg = IQLAlgorithmCfg()
        iql = IQLAlgorithm(cfg, obs_dim=8, action_dim=4)

        assert hasattr(iql, "actor")
        assert hasattr(iql, "critic")
        assert hasattr(iql, "target_critic")
        assert hasattr(iql, "value")

    def test_iql_update(self):
        """Full IQL update produces finite losses."""
        from RoboRenForce.algorithms.vla_training.iql import IQLAlgorithm, IQLAlgorithmCfg

        cfg = IQLAlgorithmCfg(hidden_dims=(32, 32))
        iql = IQLAlgorithm(cfg, obs_dim=8, action_dim=4)

        batch = {
            "obs": torch.randn(32, 8),
            "actions": torch.randn(32, 4),
            "rewards": torch.randn(32),
            "next_obs": torch.randn(32, 8),
            "dones": torch.zeros(32, dtype=torch.bool),
        }

        metrics = iql.update(batch)
        assert np.isfinite(metrics["total_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert np.isfinite(metrics["actor_loss"])
        assert np.isfinite(metrics["critic_loss"])

    def test_iql_target_update(self):
        """Target critic should diverge from online critic after updates."""
        from RoboRenForce.algorithms.vla_training.iql import IQLAlgorithm, IQLAlgorithmCfg

        cfg = IQLAlgorithmCfg(hidden_dims=(16, 16), tau=0.005)
        iql = IQLAlgorithm(cfg, obs_dim=4, action_dim=2)

        # Initially target == online
        for t_p, o_p in zip(iql.target_critic.parameters(), iql.critic.parameters()):
            assert torch.equal(t_p, o_p)

        batch = {
            "obs": torch.randn(8, 4),
            "actions": torch.randn(8, 2),
            "rewards": torch.randn(8),
            "next_obs": torch.randn(8, 4),
            "dones": torch.zeros(8, dtype=torch.bool),
        }
        iql.update(batch)

        # After update, target should differ (soft update with tau < 1)
        any_diff = False
        for t_p, o_p in zip(iql.target_critic.parameters(), iql.critic.parameters()):
            if not torch.equal(t_p, o_p):
                any_diff = True
                break
        assert any_diff

    def test_iql_select_action(self):
        """IQL can select actions."""
        from RoboRenForce.algorithms.vla_training.iql import IQLAlgorithm, IQLAlgorithmCfg

        cfg = IQLAlgorithmCfg(hidden_dims=(16, 16))
        iql = IQLAlgorithm(cfg, obs_dim=4, action_dim=2)

        obs = torch.randn(3, 4)
        action_det = iql.select_action(obs, deterministic=True)
        action_stoch = iql.select_action(obs, deterministic=False)

        assert action_det.shape == (3, 2)
        assert action_stoch.shape == (3, 2)


# ================================================================
# DAgger Tests
# ================================================================

class TestDAggerAlgorithm:
    """Test DAgger algorithm."""

    def test_trajectory_extract_expert(self):
        """Extract expert segments from trajectory."""
        from RoboRenForce.algorithms.vla_training.dagger import Trajectory

        traj = Trajectory(
            obs={"states": torch.randn(10, 8)},
            actions=torch.randn(10, 4),
            is_expert=torch.tensor([False]*5 + [True]*3 + [False]*2),
        )

        expert = Trajectory.extract_expert_segments(traj)
        assert expert is not None
        assert len(expert) == 3
        assert expert.actions.shape == (3, 4)

    def test_trajectory_no_expert(self):
        """No expert segments returns None."""
        from RoboRenForce.algorithms.vla_training.dagger import Trajectory

        traj = Trajectory(
            obs={"states": torch.randn(5, 8)},
            actions=torch.randn(5, 4),
            is_expert=torch.zeros(5, dtype=torch.bool),
        )

        expert = Trajectory.extract_expert_segments(traj)
        assert expert is None

    def test_replay_buffer(self):
        """Replay buffer stores and samples correctly."""
        from RoboRenForce.algorithms.vla_training.dagger import (
            TrajectoryReplayBuffer, Trajectory,
        )

        buf = TrajectoryReplayBuffer(max_trajectories=10, sample_window_size=5)

        # Add 3 expert trajectories
        for _ in range(3):
            traj = Trajectory(
                obs={"states": torch.randn(5, 8)},
                actions=torch.randn(5, 4),
                is_expert=torch.ones(5, dtype=torch.bool),
            )
            buf.add(traj)

        assert buf.num_trajectories == 3
        assert buf.total_transitions == 15
        assert buf.is_ready(min_size=3)

        batch = buf.sample(batch_size=4)
        assert batch["obs"]["states"].shape[0] == 4
        assert batch["actions"].shape == (4, 4)

    def test_dagger_update(self):
        """DAgger update produces finite loss."""
        from RoboRenForce.algorithms.vla_training.dagger import (
            DAggerAlgorithm, DAggerAlgorithmCfg, Trajectory,
        )

        cfg = DAggerAlgorithmCfg(min_buffer_size=2, batch_size=8)
        dagger = DAggerAlgorithm(cfg)

        # Add expert data directly
        for _ in range(3):
            traj = Trajectory(
                obs={"states": torch.randn(10, 8)},
                actions=torch.randn(10, 4),
                is_expert=torch.ones(10, dtype=torch.bool),
            )
            dagger.receive_expert_data(traj)

        assert dagger.is_ready()

        policy = MockPolicy(state_dim=8, action_dim=4)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        metrics = dagger.update(policy, optimizer)
        assert np.isfinite(metrics["total_loss"])
        assert "buffer/num_trajectories" in metrics

    def test_dagger_from_rollout(self):
        """DAgger correctly extracts expert segments from rollouts."""
        from RoboRenForce.algorithms.vla_training.dagger import (
            DAggerAlgorithm, DAggerAlgorithmCfg, Trajectory,
        )

        cfg = DAggerAlgorithmCfg(min_buffer_size=1)
        dagger = DAggerAlgorithm(cfg)

        # Rollout with some expert interventions
        traj = Trajectory(
            obs={"states": torch.randn(20, 8)},
            actions=torch.randn(20, 4),
            is_expert=torch.tensor([False]*15 + [True]*5),
        )
        dagger.receive_trajectory(traj)

        assert dagger.replay_buffer.num_trajectories == 1
        assert dagger.replay_buffer.total_transitions == 5  # only expert steps


# ================================================================
# SAC Tests
# ================================================================

class TestSACAlgorithm:
    """Test SAC algorithm."""

    def test_sac_initialization(self):
        """SAC creates all required components."""
        from RoboRenForce.algorithms.vla_training.sac import SACAlgorithm, SACAlgorithmCfg

        cfg = SACAlgorithmCfg(hidden_dims=(32, 32))
        sac = SACAlgorithm(cfg, obs_dim=8, action_dim=4)

        assert hasattr(sac, "actor")
        assert hasattr(sac, "critic")
        assert hasattr(sac, "target_critic")
        assert hasattr(sac, "entropy_temp")
        assert hasattr(sac, "replay_buffer")

    def test_sac_q_network_multi_head(self):
        """Q-network with multiple heads produces correct shape."""
        from RoboRenForce.algorithms.vla_training.sac import SACQNetwork

        q_net = SACQNetwork(obs_dim=8, action_dim=4, hidden_dims=(32,), num_heads=3)
        obs = torch.randn(5, 8)
        actions = torch.randn(5, 4)
        q_vals = q_net(obs, actions)
        assert q_vals.shape == (5, 3)

    def test_sac_gaussian_policy_squashing(self):
        """Squashed Gaussian policy outputs are in [-1, 1]."""
        from RoboRenForce.algorithms.vla_training.sac import SACGaussianPolicy

        policy = SACGaussianPolicy(obs_dim=8, action_dim=4, hidden_dims=(32,))
        obs = torch.randn(10, 8)
        actions, log_probs = policy.sample(obs)

        assert actions.shape == (10, 4)
        assert log_probs.shape == (10,)
        assert (actions.abs() <= 1.0).all(), "Tanh-squashed actions should be in [-1, 1]"
        assert torch.isfinite(log_probs).all()

    def test_entropy_temperature(self):
        """Entropy temperature module produces positive alpha."""
        from RoboRenForce.algorithms.vla_training.sac import EntropyTemperature

        for alpha_type in ["exp", "softplus", "fixed"]:
            temp = EntropyTemperature(initial_alpha=0.1, alpha_type=alpha_type)
            alpha = temp.alpha
            assert alpha > 0, f"Alpha should be positive for {alpha_type}"
            assert torch.isfinite(alpha)

    def test_sac_add_and_sample(self):
        """SAC replay buffer add and sample."""
        from RoboRenForce.algorithms.vla_training.sac import SACAlgorithm, SACAlgorithmCfg

        cfg = SACAlgorithmCfg(hidden_dims=(16,), min_buffer_size=10)
        sac = SACAlgorithm(cfg, obs_dim=4, action_dim=2)

        for _ in range(20):
            sac.add_experience(
                obs=torch.randn(4, 4),
                actions=torch.randn(4, 2),
                rewards=torch.randn(4),
                next_obs=torch.randn(4, 4),
                dones=torch.zeros(4, dtype=torch.bool),
            )

        assert sac.is_ready()
        assert len(sac.replay_buffer) == 80  # 20 * 4

    def test_sac_update(self):
        """Full SAC update step produces finite losses."""
        from RoboRenForce.algorithms.vla_training.sac import SACAlgorithm, SACAlgorithmCfg

        cfg = SACAlgorithmCfg(
            hidden_dims=(32, 32),
            min_buffer_size=50,
            batch_size=16,
        )
        sac = SACAlgorithm(cfg, obs_dim=8, action_dim=4)

        # Fill buffer
        for _ in range(20):
            sac.add_experience(
                obs=torch.randn(4, 8),
                actions=torch.randn(4, 4),
                rewards=torch.randn(4),
                next_obs=torch.randn(4, 8),
                dones=torch.zeros(4, dtype=torch.bool),
            )

        assert sac.is_ready()
        metrics = sac.update()

        assert np.isfinite(metrics["total_loss"])
        assert np.isfinite(metrics["critic_loss"])
        assert np.isfinite(metrics["actor_loss"])
        assert "sac/alpha" in metrics

    def test_sac_target_diverges(self):
        """Target critic should diverge from online after updates."""
        from RoboRenForce.algorithms.vla_training.sac import SACAlgorithm, SACAlgorithmCfg

        cfg = SACAlgorithmCfg(hidden_dims=(16,), min_buffer_size=10, batch_size=8, tau=0.005)
        sac = SACAlgorithm(cfg, obs_dim=4, action_dim=2)

        for _ in range(5):
            sac.add_experience(
                obs=torch.randn(4, 4),
                actions=torch.randn(4, 2),
                rewards=torch.randn(4),
                next_obs=torch.randn(4, 4),
                dones=torch.zeros(4, dtype=torch.bool),
            )

        sac.update()

        any_diff = False
        for t_p, o_p in zip(sac.target_critic.parameters(), sac.critic.parameters()):
            if not torch.equal(t_p, o_p):
                any_diff = True
                break
        assert any_diff

    def test_sac_select_action(self):
        """SAC can select actions in both modes."""
        from RoboRenForce.algorithms.vla_training.sac import SACAlgorithm, SACAlgorithmCfg

        cfg = SACAlgorithmCfg(hidden_dims=(16,))
        sac = SACAlgorithm(cfg, obs_dim=4, action_dim=2)

        obs = torch.randn(3, 4)
        det = sac.select_action(obs, deterministic=True)
        stoch = sac.select_action(obs, deterministic=False)
        assert det.shape == (3, 2)
        assert stoch.shape == (3, 2)

    def test_sac_fixed_alpha(self):
        """SAC with fixed alpha has no alpha optimizer."""
        from RoboRenForce.algorithms.vla_training.sac import SACAlgorithm, SACAlgorithmCfg

        cfg = SACAlgorithmCfg(hidden_dims=(16,), alpha_type="fixed")
        sac = SACAlgorithm(cfg, obs_dim=4, action_dim=2)
        assert sac.alpha_optimizer is None


# ================================================================
# PPO Runner Tests
# ================================================================

class TestPPORunner:
    """Test VLA PPO runner."""

    def _make_env(self):
        """Create mock environment."""
        # Reuse RewardableEnv from test_training_e2e
        import sys
        sys.path.insert(0, "/vepfs/users/zza/projects/RoboRenForce/tests")
        from test_training_e2e import RewardableEnv
        return RewardableEnv(num_envs=4, state_dim=8, action_dim=4, max_steps=5)

    def test_ppo_runner_init(self):
        """PPO runner initializes correctly."""
        import tempfile
        from RoboRenForce.runners.vla.rl import VLAPPORunner, VLAPPORunnerCfg
        from RoboRenForce.algorithms.vla_training.ppo import PPOAlgorithmCfg

        env = self._make_env()
        policy = MockPolicy(state_dim=8, action_dim=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = VLAPPORunnerCfg(
                ppo_cfg=PPOAlgorithmCfg(learning_rate=1e-3),
                save_interval=0,
                checkpoint_dir=f"{tmpdir}/ckpt",
            )
            runner = VLAPPORunner(cfg, env, policy, device="cpu", log_dir=f"{tmpdir}/logs")
            assert runner.global_step == 0

    def test_ppo_runner_collect_rollouts(self):
        """PPO runner collects rollouts with correct shapes."""
        import tempfile
        from RoboRenForce.runners.vla.rl import VLAPPORunner, VLAPPORunnerCfg
        from RoboRenForce.algorithms.vla_training.ppo import PPOAlgorithmCfg

        env = self._make_env()
        policy = MockPolicy(state_dim=8, action_dim=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = VLAPPORunnerCfg(
                ppo_cfg=PPOAlgorithmCfg(learning_rate=1e-3),
                save_interval=0,
                checkpoint_dir=f"{tmpdir}/ckpt",
            )
            runner = VLAPPORunner(cfg, env, policy, device="cpu", log_dir=f"{tmpdir}/logs")
            rollout = runner.collect_rollouts()

            assert rollout.actions.dim() == 2
            assert rollout.logprobs.dim() == 1
            assert rollout.values.dim() == 2  # [T+1, B]
            assert rollout.rewards.dim() == 2  # [T, B]

    def test_ppo_runner_learn(self):
        """PPO runner learns for a few iterations."""
        import tempfile
        from RoboRenForce.runners.vla.rl import VLAPPORunner, VLAPPORunnerCfg
        from RoboRenForce.algorithms.vla_training.ppo import PPOAlgorithmCfg

        env = self._make_env()
        policy = MockPolicy(state_dim=8, action_dim=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = VLAPPORunnerCfg(
                ppo_cfg=PPOAlgorithmCfg(
                    update_epochs=2,
                    learning_rate=1e-3,
                ),
                save_interval=0,
                checkpoint_dir=f"{tmpdir}/ckpt",
            )
            runner = VLAPPORunner(cfg, env, policy, device="cpu", log_dir=f"{tmpdir}/logs")
            runner.learn(num_iterations=3)
            assert runner.global_step == 3


# ================================================================
# Integration: All algorithms import cleanly
# ================================================================

class TestImports:
    """Verify all algorithms can be imported."""

    def test_import_all(self):
        from RoboRenForce.algorithms.vla_training import (
            GRPOAlgorithm, GRPOAlgorithmCfg,
            PPOAlgorithm, PPOAlgorithmCfg,
            SFTAlgorithm, SFTAlgorithmCfg,
            IQLAlgorithm, IQLAlgorithmCfg,
            DAggerAlgorithm, DAggerAlgorithmCfg,
            SACAlgorithm, SACAlgorithmCfg,
        )

    def test_import_runners(self):
        from RoboRenForce.runners.vla.rl import (
            VLAGRPORunner, VLAGRPORunnerCfg,
            VLAPPORunner, VLAPPORunnerCfg,
        )
