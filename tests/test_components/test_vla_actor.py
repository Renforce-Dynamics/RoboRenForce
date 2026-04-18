"""
Tests for VLAActor (using a mock VLM backbone for CPU testing)

IO tested:
    forward(obs_dict, deterministic) -> (B, action_horizon, action_dim)
        obs_dict: {"image": (B,C,H,W), "proprioception": (B, proprio_dim)}

    train_forward(obs_dict, target_actions) -> dict (from action head)
"""

import torch
import torch.nn as nn
import pytest

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.vlm.vlm_backbone_base import VLMBackbone, VLMBackboneCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import RegressionActionHeadCfg
from RoboRenForce.components.actor.vla_actor import VLAActor, VLAActorCfg


B = 4
IMG_C, IMG_H, IMG_W = 3, 64, 64
VL_DIM = 32
PROPRIO_DIM = 12
ACTION_DIM = 7


# ===== Mock VLM (no real model, just a linear projection) ===== #

class MockVLM(VLMBackbone):
    """Lightweight mock VLM for testing without downloading real models."""

    def __init__(self, cfg):
        super().__init__(cfg)
        # Simple conv + pool to get fixed-size features from images
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, cfg.output_dim),
        )
        self.output_dim = cfg.output_dim

    def forward(self, image, text=None, **kwargs):
        return self.encoder(image)


@configclass
class MockVLMCfg(VLMBackboneCfg):
    class_type: type = MockVLM
    model_name: str = "mock"
    output_dim: int = VL_DIM
    freeze: bool = True


# ===== Tests ===== #

class TestVLAActor:
    @pytest.fixture
    def actor(self):
        cfg = VLAActorCfg(
            vlm_backbone_cfg=MockVLMCfg(),
            freeze_vlm=True,
            fusion_cfg=FusionLayerCfg(
                output_dim=64,
                hidden_dims=[64],
            ),
            action_head_cfg=RegressionActionHeadCfg(
                action_dim=ACTION_DIM,
                action_horizon=1,
                hidden_dims=[32],
            ),
            use_proprioception=True,
        )
        dim_params = {
            "proprioception_dim": PROPRIO_DIM,
            "action_dim": ACTION_DIM,
        }
        return VLAActor(cfg, dim_params)

    @pytest.fixture
    def obs_dict(self):
        return {
            "image": torch.randn(B, IMG_C, IMG_H, IMG_W),
            "proprioception": torch.randn(B, PROPRIO_DIM),
        }

    def test_forward_shape(self, actor, obs_dict):
        out = actor(obs_dict)
        assert out.shape == (B, 1, ACTION_DIM)

    def test_vlm_frozen(self, actor):
        for p in actor.vlm.parameters():
            assert not p.requires_grad

    def test_action_head_trainable(self, actor):
        trainable = [p for p in actor.action_head.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_fusion_trainable(self, actor):
        trainable = [p for p in actor.fusion.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_train_forward(self, actor, obs_dict):
        targets = torch.randn(B, 1, ACTION_DIM)
        result = actor.train_forward(obs_dict, targets)
        assert "pred_actions" in result
        assert result["pred_actions"].shape == (B, 1, ACTION_DIM)

    def test_gradient_only_on_trainable(self, actor, obs_dict):
        targets = torch.randn(B, 1, ACTION_DIM)
        result = actor.train_forward(obs_dict, targets)
        loss = (result["pred_actions"] - targets).pow(2).mean()
        loss.backward()
        # VLM should have no gradients
        for p in actor.vlm.parameters():
            assert p.grad is None or (p.grad.abs().sum() == 0)
        # Action head should have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in actor.action_head.parameters())
        assert has_grad

    def test_get_action(self, actor, obs_dict):
        action = actor.get_action(obs_dict)
        assert action.shape == (B, 1, ACTION_DIM)


class TestVLAActorNoProprioception:
    def test_forward_without_proprio(self):
        cfg = VLAActorCfg(
            vlm_backbone_cfg=MockVLMCfg(),
            freeze_vlm=False,
            fusion_cfg=FusionLayerCfg(output_dim=64),
            action_head_cfg=RegressionActionHeadCfg(
                action_dim=ACTION_DIM,
                action_horizon=1,
                hidden_dims=[32],
            ),
            use_proprioception=False,
        )
        dim_params = {"action_dim": ACTION_DIM, "proprioception_dim": 0}
        actor = VLAActor(cfg, dim_params)
        obs_dict = {"image": torch.randn(B, 3, 64, 64)}
        out = actor(obs_dict)
        assert out.shape == (B, 1, ACTION_DIM)
