"""
Tests for VLAPretrainAlgorithm

IO tested:
    compute_loss(batch, vla_actor) -> {"total_loss": scalar, "action_loss": scalar}
    update(batch, vla_actor, optimizer) -> {"total_loss": float, "action_loss": float}
"""

import torch
import torch.nn as nn
import pytest

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.vlm.vlm_backbone_base import VLMBackbone, VLMBackboneCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import RegressionActionHeadCfg
from RoboRenForce.components.actor.action_heads.diffusion_action_head import DiffusionActionHeadCfg
from RoboRenForce.components.actor.vla_actor import VLAActor, VLAActorCfg
from RoboRenForce.algorithms.vla_training.pretrain_algorithm import (
    VLAPretrainAlgorithm,
    VLAPretrainAlgorithmCfg,
)


B = 4
VL_DIM = 32
PROPRIO_DIM = 12
ACTION_DIM = 7


# ===== Mock VLM ===== #

class _MockVLM(VLMBackbone):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.proj = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, cfg.output_dim),
        )
        self.output_dim = cfg.output_dim

    def forward(self, image, text=None, **kwargs):
        return self.proj(image)


@configclass
class _MockVLMCfg(VLMBackboneCfg):
    class_type: type = _MockVLM
    model_name: str = "mock"
    output_dim: int = VL_DIM
    freeze: bool = True


def _make_actor(action_head_cfg):
    cfg = VLAActorCfg(
        vlm_backbone_cfg=_MockVLMCfg(),
        freeze_vlm=True,
        fusion_cfg=FusionLayerCfg(output_dim=64, hidden_dims=[64]),
        action_head_cfg=action_head_cfg,
        use_proprioception=True,
    )
    return VLAActor(cfg, {"proprioception_dim": PROPRIO_DIM, "action_dim": ACTION_DIM})


def _make_batch():
    return {
        "image": torch.randn(B, 3, 64, 64),
        "proprioception": torch.randn(B, PROPRIO_DIM),
        "action": torch.randn(B, ACTION_DIM),
    }


class TestRegressionLoss:
    @pytest.fixture
    def actor(self):
        return _make_actor(RegressionActionHeadCfg(
            action_dim=ACTION_DIM, action_horizon=1, hidden_dims=[32],
        ))

    @pytest.fixture
    def algorithm(self):
        return VLAPretrainAlgorithm(VLAPretrainAlgorithmCfg(use_amp=False))

    def test_compute_loss(self, actor, algorithm):
        batch = _make_batch()
        loss_dict = algorithm.compute_loss(batch, actor)
        assert "total_loss" in loss_dict
        assert "action_loss" in loss_dict
        assert loss_dict["total_loss"].ndim == 0  # scalar

    def test_update(self, actor, algorithm):
        batch = _make_batch()
        optimizer = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad], lr=1e-3
        )
        loss_dict = algorithm.update(batch, actor, optimizer)
        assert isinstance(loss_dict["total_loss"], float)
        assert isinstance(loss_dict["action_loss"], float)

    def test_loss_decreases(self, actor, algorithm):
        """Verify that a few steps can reduce loss (basic sanity)."""
        torch.manual_seed(42)
        batch = _make_batch()
        optimizer = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad], lr=1e-3
        )
        losses = []
        for _ in range(20):
            ld = algorithm.update(batch, actor, optimizer)
            losses.append(ld["total_loss"])
        # Loss should decrease overall (not necessarily monotonically)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestDiffusionLoss:
    @pytest.fixture
    def actor(self):
        return _make_actor(DiffusionActionHeadCfg(
            action_dim=ACTION_DIM, action_horizon=1,
            num_layers=2, num_heads=4, embed_dim=64,
            num_train_steps=20, num_diffusion_steps=5,
        ))

    @pytest.fixture
    def algorithm(self):
        return VLAPretrainAlgorithm(VLAPretrainAlgorithmCfg(
            action_loss_type="diffusion", use_amp=False,
        ))

    def test_compute_loss(self, actor, algorithm):
        batch = _make_batch()
        loss_dict = algorithm.compute_loss(batch, actor)
        assert "total_loss" in loss_dict
        assert loss_dict["total_loss"].item() > 0

    def test_update(self, actor, algorithm):
        batch = _make_batch()
        optimizer = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad], lr=1e-3
        )
        loss_dict = algorithm.update(batch, actor, optimizer)
        assert isinstance(loss_dict["total_loss"], float)
