"""
Tests for all VLA model adapters — inference and fine-tuning.

Covers: Qwen2-VL, Qwen3-VL, OpenPI, GR00T, MLP Baseline.
Uses mock VLM backbones (no real model downloads) to test the full policy
pipeline: obs mapping → VLM features → fusion → action head → loss/logprobs.

Test categories:
    1. Registry & imports — all models register correctly
    2. Policy construction — cfg → policy with correct shapes
    3. Inference — predict_action, ForwardType.INFERENCE
    4. Pretrain forward — L1 loss, gradient flow through action head
    5. PPO/GRPO forward — logprobs, values, advantage-compatible
    6. SFT/SAC forward — correct dispatch
    7. Fine-tuning — freeze/unfreeze, gradient flow, parameter updates
    8. Value head — attached, shapes, gradient isolation
    9. Obs mapping — EmbodiedEnv format → VLAActor format
    10. Multi-model — same obs through different models gives valid outputs
"""

import pytest
import torch
import torch.nn as nn

from RoboRenForce.prototype.embodied import BasePolicy, ForwardType
from RoboRenForce.utils.configclass import configclass
from RoboRenForce.networks.vlm.vlm_backbone_base import VLMBackbone, VLMBackboneCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.vla_actor import VLAActorCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import RegressionActionHeadCfg
from RRF_models.modules.value_head import ValueHead
from RRF_models.registry import list_models, get_model, _MODEL_REGISTRY


# ============================================================================
# Mock VLM backbone — stands in for any real model in tests
# ============================================================================

class MockVLMForTest(VLMBackbone):
    """Lightweight VLM mock that produces features from image input."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, cfg.output_dim),
        )
        self.output_dim = cfg.output_dim

    def forward(self, image=None, text=None, images=None, **kwargs):
        img = image if image is not None else images
        if img is None:
            raise ValueError("image required")
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        return self.encoder(img)


@configclass
class MockVLMCfg(VLMBackboneCfg):
    class_type: type = MockVLMForTest
    model_name: str = "mock_vlm"
    output_dim: int = 128
    freeze: bool = False


# ============================================================================
# Shared test fixtures
# ============================================================================

B = 4            # batch size
H, W = 64, 64   # image size (small for speed)
C = 3            # channels
STATE_DIM = 14   # proprioception
ACTION_DIM = 14  # bimanual
VLM_DIM = 128    # mock VLM output dim
FUSION_DIM = 64  # fusion output dim


def make_obs(batch_size=B, device="cpu"):
    """Create a standardized EmbodiedEnv-format observation."""
    return {
        "main_images": torch.randint(0, 255, (batch_size, H, W, C),
                                     dtype=torch.uint8, device=device),
        "states": torch.randn(batch_size, STATE_DIM, device=device),
        "task_descriptions": ["test task"] * batch_size,
    }


def make_actor_cfg(freeze_vlm=False):
    """Create a VLAActorCfg with mock VLM."""
    return VLAActorCfg(
        vlm_backbone_cfg=MockVLMCfg(output_dim=VLM_DIM, freeze=freeze_vlm),
        freeze_vlm=freeze_vlm,
        fusion_cfg=FusionLayerCfg(output_dim=FUSION_DIM, hidden_dims=[FUSION_DIM]),
        action_head_cfg=RegressionActionHeadCfg(
            action_dim=ACTION_DIM, action_horizon=1, hidden_dims=[64, 64],
        ),
        use_proprioception=True,
        use_text=True,
    )


# ============================================================================
# 1. Registry & Imports
# ============================================================================

class TestRegistry:
    def test_all_models_registered(self):
        models = list_models()
        for name in ["qwen2vl", "qwen3vl", "openpi", "gr00t", "mlp_baseline"]:
            assert name in models, f"{name} not in registered models: {models}"

    def test_openpi_import(self):
        from RRF_models.openpi.openpi_policy import OpenPIPolicy, OpenPIPolicyCfg
        from RoboRenForce.networks.vlm.openpi import OpenPI, OpenPICfg
        assert OpenPICfg().model_name == "lerobot/pi05_base"

    def test_gr00t_import(self):
        from RRF_models.gr00t.gr00t_policy import GR00TPolicy, GR00TPolicyCfg
        from RoboRenForce.networks.vlm.gr00t import GR00T, GR00TCfg
        assert GR00TCfg().model_name == "nvidia/GR00T-N1.7-3B"

    def test_qwen3vl_import(self):
        from RRF_models.qwen3vl.qwen3vl_policy import Qwen3VLPolicy, Qwen3VLPolicyCfg
        from RoboRenForce.networks.vlm.qwen3vl import Qwen3VL, Qwen3VLCfg
        assert Qwen3VLCfg().output_dim == 2048

    def test_qwen2vl_import(self):
        from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicy, Qwen2VLPolicyCfg
        from RoboRenForce.networks.vlm.qwen2vl import Qwen2VLCfg
        assert Qwen2VLCfg().output_dim == 1536


# ============================================================================
# 2–3. Policy Construction & Inference — per model adapter
# ============================================================================

def _build_policy(policy_cls, cfg_cls, freeze_vlm=False, use_value_head=False):
    """Build a policy with mock VLM for any adapter."""
    cfg = cfg_cls(
        actor_cfg=make_actor_cfg(freeze_vlm=freeze_vlm),
        proprio_dim=STATE_DIM,
        use_value_head=use_value_head,
    )
    return policy_cls(cfg)


# --- Qwen2-VL ---

class TestQwen2VLPolicy:
    @pytest.fixture
    def policy(self):
        from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicy, Qwen2VLPolicyCfg
        return _build_policy(Qwen2VLPolicy, Qwen2VLPolicyCfg)

    @pytest.fixture
    def policy_with_value(self):
        from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicy, Qwen2VLPolicyCfg
        return _build_policy(Qwen2VLPolicy, Qwen2VLPolicyCfg, use_value_head=True)

    def test_isinstance_base_policy(self, policy):
        assert isinstance(policy, BasePolicy)

    def test_predict_action_shape(self, policy):
        obs = make_obs()
        actions = policy.predict_action(obs)
        assert actions.shape == (B, 1, ACTION_DIM)

    def test_predict_action_no_grad(self, policy):
        obs = make_obs()
        actions = policy.predict_action(obs)
        assert not actions.requires_grad

    def test_inference_forward(self, policy):
        obs = make_obs()
        result = policy.forward(ForwardType.INFERENCE, obs=obs)
        assert "actions" in result
        assert result["actions"].shape == (B, 1, ACTION_DIM)

    def test_pretrain_forward(self, policy):
        obs = make_obs()
        target = torch.randn(B, 1, ACTION_DIM)
        result = policy.forward(ForwardType.PRETRAIN, obs=obs, target_actions=target)
        assert "loss" in result
        assert "action_loss" in result
        assert "pred_actions" in result
        assert result["loss"].shape == ()
        assert result["loss"].item() >= 0

    def test_ppo_forward(self, policy_with_value):
        obs = make_obs()
        actions = torch.randn(B, ACTION_DIM)
        result = policy_with_value.forward(ForwardType.PPO, obs=obs, actions=actions)
        assert "logprobs" in result
        assert "pred_actions" in result
        assert "values" in result
        assert result["logprobs"].shape == (B,)
        assert result["values"].shape == (B, 1)

    def test_ppo_without_actions(self, policy_with_value):
        """PPO forward with actions=None should still return logprobs."""
        obs = make_obs()
        result = policy_with_value.forward(ForwardType.PPO, obs=obs, actions=None)
        assert "logprobs" in result
        assert result["logprobs"].shape == (B,)

    def test_value_head_exists(self, policy_with_value):
        assert policy_with_value.value_head is not None

    def test_get_value(self, policy_with_value):
        obs = make_obs()
        values = policy_with_value.get_value(obs)
        assert values is not None
        assert values.shape == (B, 1)

    def test_no_value_head_returns_none(self, policy):
        obs = make_obs()
        values = policy.get_value(obs)
        assert values is None


# --- Qwen3-VL ---

class TestQwen3VLPolicy:
    @pytest.fixture
    def policy(self):
        from RRF_models.qwen3vl.qwen3vl_policy import Qwen3VLPolicy, Qwen3VLPolicyCfg
        return _build_policy(Qwen3VLPolicy, Qwen3VLPolicyCfg)

    @pytest.fixture
    def policy_with_value(self):
        from RRF_models.qwen3vl.qwen3vl_policy import Qwen3VLPolicy, Qwen3VLPolicyCfg
        return _build_policy(Qwen3VLPolicy, Qwen3VLPolicyCfg, use_value_head=True)

    def test_predict_action(self, policy):
        actions = policy.predict_action(make_obs())
        assert actions.shape == (B, 1, ACTION_DIM)

    def test_pretrain_loss(self, policy):
        result = policy.forward(ForwardType.PRETRAIN, obs=make_obs(),
                                target_actions=torch.randn(B, 1, ACTION_DIM))
        assert result["loss"].item() >= 0

    def test_ppo_with_values(self, policy_with_value):
        result = policy_with_value.forward(ForwardType.PPO, obs=make_obs(),
                                           actions=torch.randn(B, ACTION_DIM))
        assert result["logprobs"].shape == (B,)
        assert result["values"].shape == (B, 1)


# --- OpenPI ---

class TestOpenPIPolicy:
    @pytest.fixture
    def policy(self):
        from RRF_models.openpi.openpi_policy import OpenPIPolicy, OpenPIPolicyCfg
        return _build_policy(OpenPIPolicy, OpenPIPolicyCfg)

    @pytest.fixture
    def policy_with_value(self):
        from RRF_models.openpi.openpi_policy import OpenPIPolicy, OpenPIPolicyCfg
        return _build_policy(OpenPIPolicy, OpenPIPolicyCfg, use_value_head=True)

    def test_isinstance_base_policy(self, policy):
        assert isinstance(policy, BasePolicy)

    def test_predict_action(self, policy):
        actions = policy.predict_action(make_obs())
        assert actions.shape == (B, 1, ACTION_DIM)

    def test_inference_forward(self, policy):
        result = policy.forward(ForwardType.INFERENCE, obs=make_obs())
        assert result["actions"].shape == (B, 1, ACTION_DIM)

    def test_pretrain_forward(self, policy):
        result = policy.forward(ForwardType.PRETRAIN, obs=make_obs(),
                                target_actions=torch.randn(B, 1, ACTION_DIM))
        assert "loss" in result and "pred_actions" in result
        assert result["loss"].item() >= 0

    def test_ppo_forward(self, policy_with_value):
        result = policy_with_value.forward(ForwardType.PPO, obs=make_obs(),
                                           actions=torch.randn(B, ACTION_DIM))
        assert result["logprobs"].shape == (B,)
        assert result["values"].shape == (B, 1)

    def test_sft_forward(self, policy):
        """SFT should work (delegates to pretrain path)."""
        result = policy.forward(ForwardType.SFT, obs=make_obs(),
                                target_actions=torch.randn(B, 1, ACTION_DIM))
        assert "loss" in result

    def test_sac_forward(self, policy_with_value):
        """SAC should work (delegates to logprob path)."""
        result = policy_with_value.forward(ForwardType.SAC, obs=make_obs(),
                                           actions=torch.randn(B, ACTION_DIM))
        assert "logprobs" in result

    def test_value_head(self, policy_with_value):
        values = policy_with_value.get_value(make_obs())
        assert values is not None and values.shape == (B, 1)


# --- GR00T ---

class TestGR00TPolicy:
    @pytest.fixture
    def policy(self):
        from RRF_models.gr00t.gr00t_policy import GR00TPolicy, GR00TPolicyCfg
        return _build_policy(GR00TPolicy, GR00TPolicyCfg)

    @pytest.fixture
    def policy_with_value(self):
        from RRF_models.gr00t.gr00t_policy import GR00TPolicy, GR00TPolicyCfg
        return _build_policy(GR00TPolicy, GR00TPolicyCfg, use_value_head=True)

    def test_isinstance_base_policy(self, policy):
        assert isinstance(policy, BasePolicy)

    def test_predict_action(self, policy):
        actions = policy.predict_action(make_obs())
        assert actions.shape == (B, 1, ACTION_DIM)

    def test_inference_forward(self, policy):
        result = policy.forward(ForwardType.INFERENCE, obs=make_obs())
        assert result["actions"].shape == (B, 1, ACTION_DIM)

    def test_pretrain_forward(self, policy):
        result = policy.forward(ForwardType.PRETRAIN, obs=make_obs(),
                                target_actions=torch.randn(B, 1, ACTION_DIM))
        assert "loss" in result
        assert result["loss"].item() >= 0

    def test_ppo_forward(self, policy_with_value):
        result = policy_with_value.forward(ForwardType.PPO, obs=make_obs(),
                                           actions=torch.randn(B, ACTION_DIM))
        assert result["logprobs"].shape == (B,)
        assert result["values"].shape == (B, 1)

    def test_sft_forward(self, policy):
        result = policy.forward(ForwardType.SFT, obs=make_obs(),
                                target_actions=torch.randn(B, 1, ACTION_DIM))
        assert "loss" in result

    def test_sac_forward(self, policy_with_value):
        result = policy_with_value.forward(ForwardType.SAC, obs=make_obs(),
                                           actions=torch.randn(B, ACTION_DIM))
        assert "logprobs" in result

    def test_value_head(self, policy_with_value):
        values = policy_with_value.get_value(make_obs())
        assert values is not None and values.shape == (B, 1)


# ============================================================================
# 4. Obs Mapping
# ============================================================================

class TestObsMapping:
    """Test that _map_obs correctly converts EmbodiedEnv → VLAActor format."""

    @pytest.fixture(params=["qwen2vl", "qwen3vl", "openpi", "gr00t"])
    def policy(self, request):
        if request.param == "qwen2vl":
            from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicy, Qwen2VLPolicyCfg
            return _build_policy(Qwen2VLPolicy, Qwen2VLPolicyCfg)
        elif request.param == "qwen3vl":
            from RRF_models.qwen3vl.qwen3vl_policy import Qwen3VLPolicy, Qwen3VLPolicyCfg
            return _build_policy(Qwen3VLPolicy, Qwen3VLPolicyCfg)
        elif request.param == "openpi":
            from RRF_models.openpi.openpi_policy import OpenPIPolicy, OpenPIPolicyCfg
            return _build_policy(OpenPIPolicy, OpenPIPolicyCfg)
        elif request.param == "gr00t":
            from RRF_models.gr00t.gr00t_policy import GR00TPolicy, GR00TPolicyCfg
            return _build_policy(GR00TPolicy, GR00TPolicyCfg)

    def test_image_hwc_to_chw(self, policy):
        obs = make_obs()
        mapped = policy._map_obs(obs)
        # Image should be (B, C, H, W) after mapping
        assert mapped["image"].shape == (B, C, H, W)
        assert mapped["image"].dtype in (torch.float32, torch.uint8)

    def test_proprioception_mapped(self, policy):
        obs = make_obs()
        mapped = policy._map_obs(obs)
        assert "proprioception" in mapped
        assert mapped["proprioception"].shape == (B, STATE_DIM)

    def test_text_mapped(self, policy):
        obs = make_obs()
        mapped = policy._map_obs(obs)
        assert "text" in mapped
        assert len(mapped["text"]) == B

    def test_no_states_key(self, policy):
        obs = {"main_images": torch.randint(0, 255, (B, H, W, C), dtype=torch.uint8)}
        mapped = policy._map_obs(obs)
        assert "image" in mapped
        assert "proprioception" not in mapped

    def test_float_images(self, policy):
        obs = make_obs()
        obs["main_images"] = obs["main_images"].float() / 255.0
        mapped = policy._map_obs(obs)
        assert mapped["image"].shape == (B, C, H, W)


# ============================================================================
# 5. Fine-tuning — gradient flow
# ============================================================================

class TestFineTuning:
    """Test gradient flow and parameter updates for fine-tuning."""

    @pytest.fixture(params=["qwen2vl", "qwen3vl", "openpi", "gr00t"])
    def policy_unfrozen(self, request):
        """Policy with unfrozen VLM backbone."""
        if request.param == "qwen2vl":
            from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicy, Qwen2VLPolicyCfg
            return _build_policy(Qwen2VLPolicy, Qwen2VLPolicyCfg, freeze_vlm=False)
        elif request.param == "qwen3vl":
            from RRF_models.qwen3vl.qwen3vl_policy import Qwen3VLPolicy, Qwen3VLPolicyCfg
            return _build_policy(Qwen3VLPolicy, Qwen3VLPolicyCfg, freeze_vlm=False)
        elif request.param == "openpi":
            from RRF_models.openpi.openpi_policy import OpenPIPolicy, OpenPIPolicyCfg
            return _build_policy(OpenPIPolicy, OpenPIPolicyCfg, freeze_vlm=False)
        elif request.param == "gr00t":
            from RRF_models.gr00t.gr00t_policy import GR00TPolicy, GR00TPolicyCfg
            return _build_policy(GR00TPolicy, GR00TPolicyCfg, freeze_vlm=False)

    @pytest.fixture(params=["qwen2vl", "qwen3vl", "openpi", "gr00t"])
    def policy_frozen(self, request):
        """Policy with frozen VLM backbone."""
        if request.param == "qwen2vl":
            from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicy, Qwen2VLPolicyCfg
            return _build_policy(Qwen2VLPolicy, Qwen2VLPolicyCfg, freeze_vlm=True)
        elif request.param == "qwen3vl":
            from RRF_models.qwen3vl.qwen3vl_policy import Qwen3VLPolicy, Qwen3VLPolicyCfg
            return _build_policy(Qwen3VLPolicy, Qwen3VLPolicyCfg, freeze_vlm=True)
        elif request.param == "openpi":
            from RRF_models.openpi.openpi_policy import OpenPIPolicy, OpenPIPolicyCfg
            return _build_policy(OpenPIPolicy, OpenPIPolicyCfg, freeze_vlm=True)
        elif request.param == "gr00t":
            from RRF_models.gr00t.gr00t_policy import GR00TPolicy, GR00TPolicyCfg
            return _build_policy(GR00TPolicy, GR00TPolicyCfg, freeze_vlm=True)

    def test_pretrain_gradient_flows_to_action_head(self, policy_unfrozen):
        """Gradient should flow through action head on pretrain loss."""
        policy = policy_unfrozen
        obs = make_obs()
        target = torch.randn(B, 1, ACTION_DIM)

        result = policy.forward(ForwardType.PRETRAIN, obs=obs, target_actions=target)
        result["loss"].backward()

        # Action head should have gradients
        action_head_params = list(policy.actor.action_head.parameters())
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in action_head_params)
        assert has_grad, "Action head should receive gradients"

    def test_pretrain_gradient_flows_to_vlm_when_unfrozen(self, policy_unfrozen):
        """Gradient should reach VLM backbone when unfrozen."""
        policy = policy_unfrozen
        obs = make_obs()
        target = torch.randn(B, 1, ACTION_DIM)

        result = policy.forward(ForwardType.PRETRAIN, obs=obs, target_actions=target)
        result["loss"].backward()

        vlm_params = list(policy.actor.vlm.parameters())
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in vlm_params if p.requires_grad)
        assert has_grad, "VLM should receive gradients when unfrozen"

    def test_frozen_vlm_no_gradient(self, policy_frozen):
        """VLM parameters should not receive gradients when frozen."""
        policy = policy_frozen
        obs = make_obs()
        target = torch.randn(B, 1, ACTION_DIM)

        result = policy.forward(ForwardType.PRETRAIN, obs=obs, target_actions=target)
        result["loss"].backward()

        vlm_params = list(policy.actor.vlm.parameters())
        for p in vlm_params:
            assert not p.requires_grad, "Frozen VLM params should not require grad"

    def test_frozen_vlm_action_head_still_trains(self, policy_frozen):
        """Action head should still receive gradients even with frozen VLM."""
        policy = policy_frozen
        obs = make_obs()
        target = torch.randn(B, 1, ACTION_DIM)

        result = policy.forward(ForwardType.PRETRAIN, obs=obs, target_actions=target)
        result["loss"].backward()

        action_head_params = list(policy.actor.action_head.parameters())
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in action_head_params)
        assert has_grad, "Action head should train even with frozen VLM"

    def test_optimizer_step_changes_params(self, policy_unfrozen):
        """One optimizer step should change trainable parameters."""
        policy = policy_unfrozen
        obs = make_obs()
        target = torch.randn(B, 1, ACTION_DIM)

        trainable = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-2)

        # Snapshot
        before = {id(p): p.data.clone() for p in trainable[:3]}

        result = policy.forward(ForwardType.PRETRAIN, obs=obs, target_actions=target)
        result["loss"].backward()
        optimizer.step()

        changed = sum(1 for pid, old in before.items()
                      for p in trainable[:3] if id(p) == pid
                      and not torch.equal(p.data, old))
        assert changed > 0, "At least some params should change after optimizer step"

    def test_pretrain_loss_decreases(self, policy_unfrozen):
        """Loss should decrease over a few training steps."""
        policy = policy_unfrozen
        trainable = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-3)

        obs = make_obs()
        target = torch.randn(B, 1, ACTION_DIM)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            result = policy.forward(ForwardType.PRETRAIN, obs=obs, target_actions=target)
            result["loss"].backward()
            optimizer.step()
            losses.append(result["loss"].item())

        assert losses[-1] < losses[0], (
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


# ============================================================================
# 6. PPO / GRPO fine-tuning compatibility
# ============================================================================

class TestRLFineTuning:
    """Test RL-specific forward paths for all models."""

    @pytest.fixture(params=["qwen2vl", "qwen3vl", "openpi", "gr00t"])
    def policy_rl(self, request):
        """Policy with value head enabled for RL."""
        if request.param == "qwen2vl":
            from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicy, Qwen2VLPolicyCfg
            return _build_policy(Qwen2VLPolicy, Qwen2VLPolicyCfg,
                                 freeze_vlm=True, use_value_head=True)
        elif request.param == "qwen3vl":
            from RRF_models.qwen3vl.qwen3vl_policy import Qwen3VLPolicy, Qwen3VLPolicyCfg
            return _build_policy(Qwen3VLPolicy, Qwen3VLPolicyCfg,
                                 freeze_vlm=True, use_value_head=True)
        elif request.param == "openpi":
            from RRF_models.openpi.openpi_policy import OpenPIPolicy, OpenPIPolicyCfg
            return _build_policy(OpenPIPolicy, OpenPIPolicyCfg,
                                 freeze_vlm=True, use_value_head=True)
        elif request.param == "gr00t":
            from RRF_models.gr00t.gr00t_policy import GR00TPolicy, GR00TPolicyCfg
            return _build_policy(GR00TPolicy, GR00TPolicyCfg,
                                 freeze_vlm=True, use_value_head=True)

    def test_ppo_logprobs_finite(self, policy_rl):
        obs = make_obs()
        actions = torch.randn(B, ACTION_DIM)
        result = policy_rl.forward(ForwardType.PPO, obs=obs, actions=actions)
        assert torch.isfinite(result["logprobs"]).all()

    def test_ppo_values_finite(self, policy_rl):
        obs = make_obs()
        actions = torch.randn(B, ACTION_DIM)
        result = policy_rl.forward(ForwardType.PPO, obs=obs, actions=actions)
        assert torch.isfinite(result["values"]).all()

    def test_ppo_logprobs_differentiable(self, policy_rl):
        """Logprobs should be differentiable for policy gradient."""
        obs = make_obs()
        actions = torch.randn(B, ACTION_DIM)
        result = policy_rl.forward(ForwardType.PPO, obs=obs, actions=actions)

        # Simulate GRPO/PPO loss: -mean(advantages * logprobs)
        advantages = torch.randn(B)
        loss = -(advantages * result["logprobs"]).mean()
        loss.backward()

        # Action head should get gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in policy_rl.actor.action_head.parameters())
        assert has_grad

    def test_value_head_gradient_isolated(self, policy_rl):
        """Value head gradient should not flow into VLM."""
        obs = make_obs()
        values = policy_rl.get_value(obs)
        value_loss = values.mean()
        value_loss.backward()

        # VLM should not get gradients from value head
        for p in policy_rl.actor.vlm.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() == 0, "Value head grad should not reach VLM"

    def test_grpo_compatible_rollout(self, policy_rl):
        """Simulate a GRPO-style rollout: collect → compute advantages → update."""
        obs = make_obs()

        # Collect phase
        policy_rl.eval()
        with torch.no_grad():
            result = policy_rl.forward(ForwardType.PPO, obs=obs, actions=None)
            old_logprobs = result["logprobs"].clone()
            actions = result["pred_actions"].squeeze(1)  # (B, ACTION_DIM)

        # Train phase
        policy_rl.train()
        trainable = [p for p in policy_rl.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-3)

        # Re-evaluate
        new_result = policy_rl.forward(ForwardType.PPO, obs=obs, actions=actions)
        new_logprobs = new_result["logprobs"]

        # PPO clipped surrogate loss
        ratio = torch.exp(new_logprobs - old_logprobs)
        advantages = torch.randn(B)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        loss = -torch.min(surr1, surr2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify no NaNs
        for p in trainable:
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Gradients should be finite"


# ============================================================================
# 7. Multi-model consistency
# ============================================================================

class TestMultiModel:
    """Same obs through different models should all produce valid outputs."""

    def _get_all_policies(self):
        from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicy, Qwen2VLPolicyCfg
        from RRF_models.qwen3vl.qwen3vl_policy import Qwen3VLPolicy, Qwen3VLPolicyCfg
        from RRF_models.openpi.openpi_policy import OpenPIPolicy, OpenPIPolicyCfg
        from RRF_models.gr00t.gr00t_policy import GR00TPolicy, GR00TPolicyCfg

        return {
            "qwen2vl": _build_policy(Qwen2VLPolicy, Qwen2VLPolicyCfg),
            "qwen3vl": _build_policy(Qwen3VLPolicy, Qwen3VLPolicyCfg),
            "openpi": _build_policy(OpenPIPolicy, OpenPIPolicyCfg),
            "gr00t": _build_policy(GR00TPolicy, GR00TPolicyCfg),
        }

    def test_all_models_same_obs(self):
        policies = self._get_all_policies()
        obs = make_obs()

        for name, policy in policies.items():
            actions = policy.predict_action(obs)
            assert actions.shape == (B, 1, ACTION_DIM), (
                f"{name}: expected (B, 1, ACTION_DIM), got {actions.shape}"
            )
            assert torch.isfinite(actions).all(), f"{name}: actions contain NaN/Inf"

    def test_all_models_pretrain(self):
        policies = self._get_all_policies()
        obs = make_obs()
        target = torch.randn(B, 1, ACTION_DIM)

        for name, policy in policies.items():
            result = policy.forward(ForwardType.PRETRAIN, obs=obs, target_actions=target)
            assert torch.isfinite(result["loss"]), f"{name}: loss is not finite"
            assert result["loss"].item() >= 0, f"{name}: loss is negative"


# ============================================================================
# 8. Freeze / Unfreeze
# ============================================================================

class TestFreezeUnfreeze:
    @pytest.fixture(params=["qwen2vl", "qwen3vl", "openpi", "gr00t"])
    def policy(self, request):
        if request.param == "qwen2vl":
            from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicy, Qwen2VLPolicyCfg
            return _build_policy(Qwen2VLPolicy, Qwen2VLPolicyCfg, freeze_vlm=False)
        elif request.param == "qwen3vl":
            from RRF_models.qwen3vl.qwen3vl_policy import Qwen3VLPolicy, Qwen3VLPolicyCfg
            return _build_policy(Qwen3VLPolicy, Qwen3VLPolicyCfg, freeze_vlm=False)
        elif request.param == "openpi":
            from RRF_models.openpi.openpi_policy import OpenPIPolicy, OpenPIPolicyCfg
            return _build_policy(OpenPIPolicy, OpenPIPolicyCfg, freeze_vlm=False)
        elif request.param == "gr00t":
            from RRF_models.gr00t.gr00t_policy import GR00TPolicy, GR00TPolicyCfg
            return _build_policy(GR00TPolicy, GR00TPolicyCfg, freeze_vlm=False)

    def test_freeze_disables_grad(self, policy):
        """freeze_backbone() should disable requires_grad on VLM params."""
        policy.freeze_backbone()
        for p in policy.actor.vlm.parameters():
            assert not p.requires_grad

    def test_unfreeze_enables_grad(self, policy):
        """unfreeze_backbone() should re-enable requires_grad."""
        policy.freeze_backbone()
        policy.unfreeze_backbone()
        has_trainable = any(p.requires_grad for p in policy.actor.vlm.parameters())
        assert has_trainable

    def test_trainable_param_count_changes(self, policy):
        """Freezing should reduce trainable param count."""
        total_before = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        policy.freeze_backbone()
        total_after = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        assert total_after < total_before


# ============================================================================
# 9. Checkpoint save/load round-trip
# ============================================================================

class TestCheckpoint:
    @pytest.fixture(params=["openpi", "gr00t"])
    def policy(self, request):
        if request.param == "openpi":
            from RRF_models.openpi.openpi_policy import OpenPIPolicy, OpenPIPolicyCfg
            return _build_policy(OpenPIPolicy, OpenPIPolicyCfg)
        elif request.param == "gr00t":
            from RRF_models.gr00t.gr00t_policy import GR00TPolicy, GR00TPolicyCfg
            return _build_policy(GR00TPolicy, GR00TPolicyCfg)

    def test_state_dict_round_trip(self, policy, tmp_path):
        """Save and load state_dict should preserve parameters."""
        obs = make_obs()
        actions_before = policy.predict_action(obs)

        # Save
        path = tmp_path / "checkpoint.pt"
        torch.save({"policy_state_dict": policy.state_dict()}, path)

        # Perturb
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        actions_perturbed = policy.predict_action(obs)
        assert not torch.allclose(actions_before, actions_perturbed, atol=1e-3)

        # Load
        ckpt = torch.load(path, weights_only=False)
        policy.load_state_dict(ckpt["policy_state_dict"])

        actions_after = policy.predict_action(obs)
        assert torch.allclose(actions_before, actions_after, atol=1e-6)
