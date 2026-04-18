"""
Tests for VLAPretrainRunner (end-to-end single-GPU)

IO tested:
    __init__(cfg, log_dir, device) -> runner with dataset, actor, algorithm
    learn(num_epochs) -> trains and saves checkpoints
    validate() -> float (avg val loss)
    save_checkpoint() / load_checkpoint() -> persist/restore state
"""

import torch
import torch.nn as nn
import pytest

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.vlm.vlm_backbone_base import VLMBackbone, VLMBackboneCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import RegressionActionHeadCfg
from RoboRenForce.components.actor.vla_actor import VLAActorCfg
from RoboRenForce.algorithms.vla_training.pretrain_algorithm import VLAPretrainAlgorithmCfg
from RoboRenForce.runners.vla.pretrain.vla_pretrain_runner import (
    VLAPretrainRunner,
    VLAPretrainRunnerCfg,
)


VL_DIM = 32
PROPRIO_DIM = 12
ACTION_DIM = 7
NUM_FRAMES = 100


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


# ===== Minimal Dataset ===== #

class _DummyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg=None):
        self.n = NUM_FRAMES

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 64, 64),
            "observation.state": torch.randn(PROPRIO_DIM),
            "proprioception": torch.randn(PROPRIO_DIM),
            "action": torch.randn(ACTION_DIM),
        }


@configclass
class _DummyDatasetCfg(ModuleBaseCfg):
    class_type: type = _DummyDataset


# ===== Tests ===== #

class TestVLAPretrainRunner:
    @pytest.fixture
    def runner(self, tmp_path):
        cfg = VLAPretrainRunnerCfg(
            vla_actor_cfg=VLAActorCfg(
                vlm_backbone_cfg=_MockVLMCfg(),
                freeze_vlm=True,
                fusion_cfg=FusionLayerCfg(output_dim=64, hidden_dims=[64]),
                action_head_cfg=RegressionActionHeadCfg(
                    action_dim=ACTION_DIM,
                    action_horizon=1,
                    hidden_dims=[32],
                ),
                use_proprioception=True,
            ),
            algorithm_cfg=VLAPretrainAlgorithmCfg(
                learning_rate=1e-3,
                warmup_steps=5,
                use_amp=False,
            ),
            dataset_cfg=_DummyDatasetCfg(),
            batch_size=8,
            num_epochs=2,
            num_workers=0,
            log_interval=5,
            save_interval=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        return VLAPretrainRunner(
            cfg,
            log_dir=str(tmp_path / "logs"),
            device="cpu",
        )

    def test_init(self, runner):
        assert runner.vla_actor is not None
        assert runner.algorithm is not None
        assert runner.optimizer is not None
        assert len(runner.train_dataset) > 0

    def test_learn(self, runner):
        runner.learn(num_epochs=2)
        assert runner.global_step > 0

    def test_validate(self, runner):
        val_loss = runner.validate()
        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_checkpoint_roundtrip(self, runner, tmp_path):
        runner.learn(num_epochs=1)
        step_before = runner.global_step

        runner.save_checkpoint(tag="test")
        ckpt_path = tmp_path / "ckpt" / "checkpoint_test.pt"
        assert ckpt_path.exists()

        runner.global_step = 0
        runner.load_checkpoint(str(ckpt_path))
        assert runner.global_step == step_before
