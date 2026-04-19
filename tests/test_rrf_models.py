"""
Unit tests for RRF_models components.

Tests BasePolicy/ForwardType, MLPBaselinePolicy, Model Registry, and ValueHead.
Pure Python tests — no GPU or pretrained model loading needed.
"""

import pytest
import torch

from RoboRenForce.prototype.embodied.base_policy import BasePolicy, ForwardType
from RRF_models.mlp_baseline.mlp_policy import MLPBaselinePolicy
from RRF_models.registry import register_model, get_model, list_models, _MODEL_REGISTRY
from RRF_models.modules.value_head import ValueHead


# ---- 1. BasePolicy + ForwardType ----

class TestForwardType:
    def test_enum_values(self):
        expected = {"INFERENCE", "PRETRAIN", "SFT", "PPO", "SAC"}
        actual = {e.name for e in ForwardType}
        assert actual == expected

    def test_enum_string_values(self):
        assert ForwardType.INFERENCE.value == "inference"
        assert ForwardType.PRETRAIN.value == "pretrain"
        assert ForwardType.SFT.value == "sft"
        assert ForwardType.PPO.value == "ppo"
        assert ForwardType.SAC.value == "sac"


class TestBasePolicy:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BasePolicy()


# ---- 2. MLPBaselinePolicy ----

class TestMLPBaselinePolicy:
    @pytest.fixture
    def policy(self):
        return MLPBaselinePolicy(state_dim=16, action_dim=7, action_horizon=1)

    def test_predict_action_shape(self, policy):
        obs = {"states": torch.randn(2, 16)}
        actions = policy.predict_action(obs)
        assert actions.shape == (2, 1, 7), f"Expected (2, 1, 7), got {actions.shape}"

    def test_predict_action_no_grad(self, policy):
        obs = {"states": torch.randn(2, 16)}
        actions = policy.predict_action(obs)
        assert not actions.requires_grad

    def test_pretrain_forward(self, policy):
        obs = {"states": torch.randn(2, 16)}
        target_actions = torch.randn(2, 1, 7)
        result = policy.pretrain_forward(obs=obs, target_actions=target_actions)
        assert "loss" in result
        assert "action_loss" in result
        assert "pred_actions" in result
        assert result["pred_actions"].shape == (2, 1, 7)
        assert result["loss"].shape == ()  # scalar

    def test_forward_pretrain_mode(self, policy):
        obs = {"states": torch.randn(2, 16)}
        target_actions = torch.randn(2, 1, 7)
        result = policy.forward(
            forward_type=ForwardType.PRETRAIN,
            obs=obs,
            target_actions=target_actions,
        )
        assert "loss" in result

    def test_forward_inference_mode(self, policy):
        obs = {"states": torch.randn(2, 16)}
        result = policy.forward(forward_type=ForwardType.INFERENCE, obs=obs)
        assert "actions" in result
        assert result["actions"].shape == (2, 1, 7)


# ---- 3. Model Registry ----

class TestModelRegistry:
    def test_register_and_get(self):
        # Register a dummy model
        def dummy_builder(cfg=None, **kwargs):
            return MLPBaselinePolicy(state_dim=8, action_dim=4)

        # Use a unique name to avoid collision
        name = "_test_dummy_model"
        if name in _MODEL_REGISTRY:
            del _MODEL_REGISTRY[name]
        register_model(name, dummy_builder)
        try:
            model = get_model(name)
            assert isinstance(model, MLPBaselinePolicy)
        finally:
            del _MODEL_REGISTRY[name]

    def test_get_mlp_baseline(self):
        model = get_model("mlp_baseline", cfg={"state_dim": 16, "action_dim": 7})
        assert isinstance(model, MLPBaselinePolicy)

    def test_list_models_includes_mlp_baseline(self):
        models = list_models()
        assert "mlp_baseline" in models

    def test_get_unknown_model_raises(self):
        with pytest.raises(KeyError):
            get_model("nonexistent_model_xyz")


# ---- 4. ValueHead ----

class TestValueHead:
    def test_output_shape(self):
        head = ValueHead(input_dim=64)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"

    def test_output_shape_batch1(self):
        head = ValueHead(input_dim=32, hidden_dims=(64,))
        x = torch.randn(1, 32)
        out = head(x)
        assert out.shape == (1, 1)

    def test_gradient_flows(self):
        head = ValueHead(input_dim=16)
        x = torch.randn(2, 16, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None
