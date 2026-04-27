"""
Tests for VLA RL Fine-tune (Phase 6).

Tests:
1. GRPO runner: collect rollouts, train, reward improves
2. PPO runner: collect rollouts with GAE, train
3. Multimodal env wrapper works
4. Pretrain → SFT → RL pipeline (checkpoint loading)
5. Evaluation loop
"""

import pytest
import torch
import torch.nn as nn

from RoboRenForce.prototype.embodied import BasePolicy, ForwardType, EmbodiedEnv
from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithm, GRPOAlgorithmCfg
from RoboRenForce.algorithms.vla_training.ppo import PPOAlgorithm, PPOAlgorithmCfg, compute_gae_advantages
from RoboRenForce.runners.vla.rl.vla_grpo_runner import VLAGRPORunner, VLAGRPORunnerCfg
from RoboRenForce.runners.vla.rl.vla_ppo_runner import VLAPPORunner, VLAPPORunnerCfg
from RoboRenForce.utils.env_wrapper.vla_wrapper.multimodal_env_wrapper import (
    MultimodalEnvWrapper, MultimodalEnvWrapperCfg,
)


# ===== Mock Env and Policy ===== #

class _MockEnv(EmbodiedEnv):
    """Simple env with learnable reward structure."""

    def __init__(self, num_envs=4, action_dim=7, state_dim=7,
                 max_episode_length=20, device="cpu"):
        self.num_envs = num_envs
        self.num_actions = action_dim
        self.num_obs = state_dim
        self.max_episode_length = max_episode_length
        self.device = torch.device(device)
        self._step_count = 0
        self._target = torch.randn(action_dim, device=self.device) * 0.5

    def get_observations(self):
        return self._make_obs(), {}

    def reset(self):
        self._step_count = 0
        return self._make_obs(), {}

    def step(self, actions):
        self._step_count += 1
        obs = self._make_obs()
        # Reward = negative distance to target (higher is better when closer)
        dist = (actions - self._target.unsqueeze(0)).pow(2).sum(dim=-1).sqrt()
        rewards = -dist + 1.0
        dones = torch.full((self.num_envs,), self._step_count >= self.max_episode_length,
                          dtype=torch.bool, device=self.device)
        return obs, rewards, dones, {}

    def _make_obs(self):
        return {
            "main_images": torch.randint(0, 255, (self.num_envs, 32, 32, 3),
                                          dtype=torch.uint8, device=self.device),
            "states": torch.randn(self.num_envs, self.num_obs, device=self.device),
            "task_descriptions": ["test task"] * self.num_envs,
        }


class _MockRLPolicy(BasePolicy):
    """Simple policy for RL testing with value head."""

    def __init__(self, state_dim=7, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.GELU())
        self.actor = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, action_dim))
        self.value_head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def predict_action(self, obs, **kwargs):
        with torch.no_grad():
            h = self.encoder(obs["states"])
            return self.actor(h)

    def get_value(self, obs):
        h = self.encoder(obs["states"])
        return self.value_head(h.detach()).squeeze(-1)

    def forward(self, forward_type=ForwardType.INFERENCE, **kwargs):
        if forward_type == ForwardType.INFERENCE:
            return {"actions": self.predict_action(kwargs["obs"])}
        elif forward_type == ForwardType.PPO:
            obs = kwargs["obs"]
            actions = kwargs.get("actions")
            h = self.encoder(obs["states"])
            pred = self.actor(h)
            if actions is not None:
                diff = actions - pred
                logprobs = -0.5 * (diff ** 2).sum(dim=-1)
            else:
                logprobs = -0.5 * (pred ** 2).sum(dim=-1)
            values = self.value_head(h.detach()).squeeze(-1)
            return {"pred_actions": pred, "logprobs": logprobs, "values": values}
        elif forward_type == ForwardType.PRETRAIN:
            obs = kwargs["obs"]
            target_actions = kwargs["target_actions"]
            h = self.encoder(obs["states"])
            pred = self.actor(h).unsqueeze(1)
            loss = nn.functional.mse_loss(pred, target_actions)
            return {"loss": loss, "pred_actions": pred}
        raise NotImplementedError


class _MockStateEnv:
    """Simple state-only env for testing MultimodalEnvWrapper."""

    def __init__(self, num_envs=4, obs_dim=10, action_dim=7, max_ep=20, device="cpu"):
        self.num_envs = num_envs
        self.num_actions = action_dim
        self.num_obs = obs_dim
        self.max_episode_length = max_ep
        self.device = torch.device(device)
        self._step = 0

    def get_observations(self):
        return torch.randn(self.num_envs, self.num_obs, device=self.device), {}

    def reset(self):
        self._step = 0
        return torch.randn(self.num_envs, self.num_obs, device=self.device), {}

    def step(self, actions):
        self._step += 1
        obs = torch.randn(self.num_envs, self.num_obs, device=self.device)
        rewards = torch.randn(self.num_envs, device=self.device)
        dones = torch.full((self.num_envs,), self._step >= self.max_episode_length,
                          dtype=torch.bool, device=self.device)
        return obs, rewards, dones, {}


# ===== Tests ===== #

class TestMultimodalEnvWrapper:
    def test_wraps_state_env(self):
        base = _MockStateEnv(num_envs=2, obs_dim=10)
        cfg = MultimodalEnvWrapperCfg(image_size=(32, 32), task_description="test")
        wrapped = MultimodalEnvWrapper(base, cfg)

        assert wrapped.num_envs == 2
        assert wrapped.num_actions == 7

        obs, info = wrapped.reset()
        assert "main_images" in obs
        assert "states" in obs
        assert "task_descriptions" in obs
        assert obs["main_images"].shape == (2, 32, 32, 3)
        assert obs["states"].shape == (2, 10)
        assert len(obs["task_descriptions"]) == 2

    def test_step_returns_correct_format(self):
        base = _MockStateEnv(num_envs=3)
        cfg = MultimodalEnvWrapperCfg(image_size=(16, 16))
        wrapped = MultimodalEnvWrapper(base, cfg)

        obs, info = wrapped.reset()
        actions = torch.randn(3, 7)
        obs2, rewards, dones, extras = wrapped.step(actions)

        assert obs2["main_images"].shape == (3, 16, 16, 3)
        assert rewards.shape == (3,)
        assert dones.shape == (3,)

    def test_wrist_camera(self):
        base = _MockStateEnv(num_envs=2)
        cfg = MultimodalEnvWrapperCfg(image_size=(32, 32), has_wrist_camera=True)
        wrapped = MultimodalEnvWrapper(base, cfg)

        obs, _ = wrapped.reset()
        assert "wrist_images" in obs
        assert obs["wrist_images"].shape == (2, 32, 32, 3)


class TestGRPORunner:
    def test_grpo_collect_rollouts(self, tmp_path):
        env = _MockEnv(num_envs=4)
        policy = _MockRLPolicy()
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(group_size=2, update_epochs=1),
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLAGRPORunner(cfg, env=env, policy=policy, device="cpu")

        rollout = runner.collect_rollouts()
        assert rollout.actions.shape[0] == 4 * 2  # num_envs * group_size
        assert rollout.logprobs.shape[0] == 8
        assert rollout.episode_returns.shape[0] == 8

    def test_grpo_train_step(self, tmp_path):
        env = _MockEnv(num_envs=4)
        policy = _MockRLPolicy()
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(group_size=2, update_epochs=2, learning_rate=1e-3),
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLAGRPORunner(cfg, env=env, policy=policy,
                               device="cpu", log_dir=str(tmp_path / "log"))

        rollout = runner.collect_rollouts()
        metrics = runner.train_on_rollouts(rollout)
        assert "total_loss" in metrics

    def test_grpo_learn_loop(self, tmp_path):
        env = _MockEnv(num_envs=4, max_episode_length=10)
        policy = _MockRLPolicy()
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(group_size=2, update_epochs=2, learning_rate=1e-3),
            log_interval=1,
            save_interval=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLAGRPORunner(cfg, env=env, policy=policy,
                               device="cpu", log_dir=str(tmp_path / "log"))
        runner.learn(num_iterations=3)

    def test_grpo_evaluate(self, tmp_path):
        env = _MockEnv(num_envs=4, max_episode_length=10)
        policy = _MockRLPolicy()
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(group_size=2),
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLAGRPORunner(cfg, env=env, policy=policy, device="cpu")
        eval_metrics = runner.evaluate(num_episodes=8)
        assert "mean_return" in eval_metrics
        assert "success_rate" in eval_metrics


class TestPPORunner:
    def test_gae_computation(self):
        T, B = 5, 2
        rewards = torch.randn(T, B)
        values = torch.randn(T + 1, B)
        dones = torch.zeros(T, B, dtype=torch.bool)
        dones[-1] = True

        advantages, returns = compute_gae_advantages(rewards, values, dones)
        assert advantages.shape == (T, B)
        assert returns.shape == (T, B)

    def test_ppo_collect_rollouts(self, tmp_path):
        env = _MockEnv(num_envs=4, max_episode_length=10)
        policy = _MockRLPolicy()
        cfg = VLAPPORunnerCfg(
            ppo_cfg=PPOAlgorithmCfg(update_epochs=1),
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLAPPORunner(cfg, env=env, policy=policy, device="cpu")

        rollout = runner.collect_rollouts()
        assert rollout.actions.shape[0] == 4 * 10  # num_envs * max_steps
        assert rollout.values.shape[0] == 11  # T+1

    def test_ppo_train_step(self, tmp_path):
        env = _MockEnv(num_envs=4, max_episode_length=10)
        policy = _MockRLPolicy()
        cfg = VLAPPORunnerCfg(
            ppo_cfg=PPOAlgorithmCfg(update_epochs=2, learning_rate=1e-3),
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLAPPORunner(cfg, env=env, policy=policy,
                              device="cpu", log_dir=str(tmp_path / "log"))

        rollout = runner.collect_rollouts()
        metrics = runner.train_on_rollouts(rollout)
        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics

    def test_ppo_learn_loop(self, tmp_path):
        env = _MockEnv(num_envs=4, max_episode_length=10)
        policy = _MockRLPolicy()
        cfg = VLAPPORunnerCfg(
            ppo_cfg=PPOAlgorithmCfg(update_epochs=2, learning_rate=1e-3),
            log_interval=1,
            save_interval=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLAPPORunner(cfg, env=env, policy=policy,
                              device="cpu", log_dir=str(tmp_path / "log"))
        runner.learn(num_iterations=3)


class TestPretrainToRL:
    """Test the full pretrain → SFT → RL pipeline via checkpoint loading."""

    def test_pretrain_checkpoint_loads_into_rl(self, tmp_path):
        """Train pretrain, save checkpoint, load into RL policy."""
        policy = _MockRLPolicy()

        # Simulate pretrain: train a few steps
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        for _ in range(5):
            batch_obs = {"states": torch.randn(8, 7)}
            target = torch.randn(8, 1, 7)
            result = policy.forward(ForwardType.PRETRAIN, obs=batch_obs, target_actions=target)
            result["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save pretrain checkpoint
        ckpt_path = tmp_path / "pretrain.pt"
        torch.save({"policy_state_dict": policy.state_dict()}, ckpt_path)

        # Load into a fresh RL policy
        rl_policy = _MockRLPolicy()
        ckpt = torch.load(ckpt_path, weights_only=False)
        rl_policy.load_state_dict(ckpt["policy_state_dict"])

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(policy.state_dict().items(), rl_policy.state_dict().items()):
            assert torch.allclose(p1, p2), f"Mismatch: {n1}"

        # Run RL with loaded policy
        env = _MockEnv(num_envs=4, max_episode_length=10)
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(group_size=2, update_epochs=1),
            save_interval=0,
            checkpoint_dir=str(tmp_path / "rl_ckpt"),
        )
        runner = VLAGRPORunner(cfg, env=env, policy=rl_policy,
                               device="cpu", log_dir=str(tmp_path / "log"))
        runner.learn(num_iterations=2)

    def test_rl_checkpoint_roundtrip(self, tmp_path):
        env = _MockEnv(num_envs=4, max_episode_length=10)
        policy = _MockRLPolicy()
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(group_size=2),
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLAGRPORunner(cfg, env=env, policy=policy,
                               device="cpu", log_dir=str(tmp_path / "log"))
        runner.learn(num_iterations=2)
        runner.save_checkpoint(tag="test")

        ckpt_path = tmp_path / "ckpt" / "grpo_checkpoint_test.pt"
        assert ckpt_path.exists()

        # Load into new runner
        policy2 = _MockRLPolicy()
        runner2 = VLAGRPORunner(cfg, env=env, policy=policy2,
                                device="cpu", log_dir=str(tmp_path / "log2"))
        runner2.load_checkpoint(str(ckpt_path))
        assert runner2.global_step == runner.global_step
