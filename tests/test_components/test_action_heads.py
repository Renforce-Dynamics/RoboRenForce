"""
Tests for RegressionActionHead and DiffusionActionHead

IO tested:
    RegressionActionHead:
        forward(features, deterministic) -> (B, action_horizon, action_dim)
        train_forward(features, targets) -> {"pred_actions", "target_actions"}

    DiffusionActionHead:
        train_forward(features, targets) -> {"noise_pred", "noise_target", "timesteps"}
        sample_actions(features, num_steps) -> (B, action_horizon, action_dim)
"""

import torch
import pytest

from RoboRenForce.components.actor.action_heads.regression_action_head import (
    RegressionActionHead,
    RegressionActionHeadCfg,
)
from RoboRenForce.components.actor.action_heads.diffusion_action_head import (
    DiffusionActionHead,
    DiffusionActionHeadCfg,
)


B = 4
INPUT_DIM = 128
ACTION_DIM = 7
ACTION_HORIZON = 1


# ===== Regression Action Head ===== #

class TestRegressionActionHead:
    @pytest.fixture
    def head(self):
        cfg = RegressionActionHeadCfg(
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
            hidden_dims=[64, 64],
        )
        return RegressionActionHead(cfg, {"input_dim": INPUT_DIM})

    def test_output_shape(self, head):
        features = torch.randn(B, INPUT_DIM)
        out = head(features)
        assert out.shape == (B, ACTION_HORIZON, ACTION_DIM)

    def test_multi_horizon(self):
        cfg = RegressionActionHeadCfg(
            action_dim=ACTION_DIM,
            action_horizon=4,
            hidden_dims=[64],
        )
        head = RegressionActionHead(cfg, {"input_dim": INPUT_DIM})
        out = head(torch.randn(B, INPUT_DIM))
        assert out.shape == (B, 4, ACTION_DIM)

    def test_train_forward(self, head):
        features = torch.randn(B, INPUT_DIM)
        targets = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
        result = head.train_forward(features, targets)
        assert "pred_actions" in result
        assert "target_actions" in result
        assert result["pred_actions"].shape == (B, ACTION_HORIZON, ACTION_DIM)
        assert torch.equal(result["target_actions"], targets)

    def test_gradient_flow(self, head):
        features = torch.randn(B, INPUT_DIM, requires_grad=True)
        out = head(features)
        out.sum().backward()
        assert features.grad is not None


# ===== Diffusion Action Head ===== #

class TestDiffusionActionHead:
    @pytest.fixture
    def head(self):
        cfg = DiffusionActionHeadCfg(
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
            num_layers=2,
            num_heads=4,
            embed_dim=64,
            num_train_steps=50,
            num_diffusion_steps=5,
        )
        return DiffusionActionHead(cfg, {"input_dim": INPUT_DIM})

    def test_noise_schedule_registered(self, head):
        assert hasattr(head, "alpha_bar")
        assert head.alpha_bar.shape == (50,)
        assert head.alpha_bar[0] > head.alpha_bar[-1]  # decreasing

    def test_train_forward(self, head):
        head.train()
        features = torch.randn(B, INPUT_DIM)
        targets = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
        result = head.train_forward(features, targets)
        assert "noise_pred" in result
        assert "noise_target" in result
        assert "timesteps" in result
        assert result["noise_pred"].shape == (B, ACTION_HORIZON, ACTION_DIM)
        assert result["noise_target"].shape == (B, ACTION_HORIZON, ACTION_DIM)

    def test_train_forward_gradient(self, head):
        head.train()
        features = torch.randn(B, INPUT_DIM, requires_grad=True)
        targets = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
        result = head.train_forward(features, targets)
        loss = (result["noise_pred"] - result["noise_target"]).pow(2).mean()
        loss.backward()
        assert features.grad is not None

    def test_sample_actions(self, head):
        head.eval()
        features = torch.randn(B, INPUT_DIM)
        actions = head.sample_actions(features, num_steps=5)
        assert actions.shape == (B, ACTION_HORIZON, ACTION_DIM)

    def test_forward_eval_mode(self, head):
        head.eval()
        features = torch.randn(B, INPUT_DIM)
        actions = head(features)
        assert actions.shape == (B, ACTION_HORIZON, ACTION_DIM)

    def test_forward_train_mode_raises(self, head):
        head.train()
        features = torch.randn(B, INPUT_DIM)
        with pytest.raises(RuntimeError, match="train_forward"):
            head(features)

    def test_multi_horizon(self):
        cfg = DiffusionActionHeadCfg(
            action_dim=ACTION_DIM,
            action_horizon=4,
            num_layers=2,
            num_heads=4,
            embed_dim=64,
            num_train_steps=20,
            num_diffusion_steps=5,
        )
        head = DiffusionActionHead(cfg, {"input_dim": INPUT_DIM})
        head.eval()
        out = head.sample_actions(torch.randn(B, INPUT_DIM), num_steps=5)
        assert out.shape == (B, 4, ACTION_DIM)
