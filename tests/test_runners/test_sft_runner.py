"""
Tests for VLA SFT Runner (Phase 5).

Tests:
1. SFT runner trains and loss decreases
2. KL regularization works
3. Freeze backbone reduces trainable params
4. Checkpoint save/load round-trip
5. Pretrained checkpoint loading
"""

import copy
import pytest
import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBaseCfg
from RoboRenForce.prototype.embodied import BasePolicy, ForwardType
from RoboRenForce.algorithms.vla_training.sft import SFTAlgorithm, SFTAlgorithmCfg
from RoboRenForce.runners.vla.post_train.sft_runner import VLASFTRunner, VLASFTRunnerCfg


# ===== Fixtures ===== #

class _MockPolicy(BasePolicy):
    """Simple policy for testing."""

    def __init__(self, state_dim=16, action_dim=7, action_horizon=1):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, action_dim * action_horizon),
        )

    def predict_action(self, obs, **kwargs):
        with torch.no_grad():
            h = self.backbone(obs["states"])
            raw = self.action_head(h)
        return raw.view(-1, self.action_horizon, self.action_dim)

    def forward(self, forward_type=ForwardType.PRETRAIN, **kwargs):
        if forward_type in (ForwardType.PRETRAIN, ForwardType.SFT):
            return self.pretrain_forward(**kwargs)
        elif forward_type == ForwardType.INFERENCE:
            return {"actions": self.predict_action(kwargs["obs"])}
        raise NotImplementedError

    def pretrain_forward(self, obs, target_actions, **kwargs):
        h = self.backbone(obs["states"])
        raw = self.action_head(h)
        pred = raw.view(-1, self.action_horizon, self.action_dim)
        loss = nn.functional.mse_loss(pred, target_actions)
        return {"loss": loss, "action_loss": loss, "pred_actions": pred}

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)


class _MockDataset(torch.utils.data.Dataset):
    def __init__(self, cfg=None, n=100, state_dim=16, action_dim=7):
        self.n = n
        torch.manual_seed(42)
        self.states = torch.randn(n, state_dim)
        self.actions = torch.sin(self.states[:, :action_dim]) * 0.5

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"states": self.states[idx], "action": self.actions[idx]}


@configclass
class _MockDatasetCfg(ModuleBaseCfg):
    class_type: type = _MockDataset


_MOCK_REGISTRY = {"test_mock_sft": _MockPolicy}


def _install_mock_registry():
    """Install mock model registry."""
    import sys
    import types
    if "RRF_models" not in sys.modules:
        mod = types.ModuleType("RRF_models")
        mod.get_model = lambda name, cfg=None, **kw: _MOCK_REGISTRY[name]()
        sys.modules["RRF_models"] = mod
    else:
        import RRF_models
        _orig = getattr(RRF_models, "_orig_get_model", RRF_models.get_model)
        RRF_models._orig_get_model = _orig
        RRF_models.get_model = lambda name, cfg=None, **kw: (
            _MOCK_REGISTRY[name]() if name in _MOCK_REGISTRY else _orig(name, cfg=cfg, **kw)
        )


@pytest.fixture(autouse=True)
def setup_registry():
    _install_mock_registry()


# ===== Tests ===== #

class TestSFTAlgorithm:
    """Test the SFT algorithm directly."""

    def test_sft_compute_loss(self):
        algo = SFTAlgorithmCfg(action_loss_type="mse", kl_coef=0.0).construct_from_cfg()
        policy = _MockPolicy()

        batch = {
            "states": torch.randn(8, 16),
            "action": torch.randn(8, 7),
        }
        loss_dict = algo.compute_loss(batch, policy)
        assert "total_loss" in loss_dict
        assert "action_loss" in loss_dict
        assert loss_dict["total_loss"].item() > 0

    def test_sft_with_kl(self):
        algo = SFTAlgorithmCfg(kl_coef=0.1).construct_from_cfg()
        policy = _MockPolicy()
        ref_policy = copy.deepcopy(policy)
        algo.set_reference_policy(ref_policy)

        batch = {
            "states": torch.randn(8, 16),
            "action": torch.randn(8, 7),
        }
        loss_dict = algo.compute_loss(batch, policy)
        assert "total_loss" in loss_dict

    def test_sft_update_step(self):
        algo = SFTAlgorithmCfg(learning_rate=1e-3).construct_from_cfg()
        policy = _MockPolicy()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

        batch = {
            "states": torch.randn(8, 16),
            "action": torch.randn(8, 7),
        }
        loss1 = algo.update(batch, policy, optimizer)
        loss2 = algo.update(batch, policy, optimizer)
        assert loss2["total_loss"] <= loss1["total_loss"] * 1.5  # Not diverging

    def test_sft_l1_loss(self):
        algo = SFTAlgorithmCfg(action_loss_type="l1").construct_from_cfg()
        policy = _MockPolicy()
        batch = {
            "states": torch.randn(4, 16),
            "action": torch.randn(4, 7),
        }
        loss_dict = algo.compute_loss(batch, policy)
        assert loss_dict["total_loss"].item() > 0


class TestVLASFTRunner:
    """Test the full SFT runner."""

    def test_sft_runner_mock_trains(self, tmp_path):
        cfg = VLASFTRunnerCfg(
            model_type="test_mock_sft",
            dataset_cfg=_MockDatasetCfg(),
            freeze_backbone=False,
            learning_rate=1e-3,
            batch_size=16,
            num_epochs=5,
            num_workers=0,
            log_interval=5,
            save_interval=0,  # disable auto-save
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLASFTRunner(cfg, log_dir=str(tmp_path / "log"), device="cpu")

        # Record initial loss
        init_batch = next(iter(runner.train_loader))
        init_loss = runner.algorithm.compute_loss(init_batch, runner.policy)["total_loss"].item()

        runner.learn(num_epochs=5)

        # Loss should decrease
        final_batch = next(iter(runner.train_loader))
        final_loss = runner.algorithm.compute_loss(final_batch, runner.policy)["total_loss"].item()
        assert final_loss < init_loss * 0.95, f"Loss did not decrease: {init_loss:.4f} -> {final_loss:.4f}"

    def test_sft_runner_with_kl(self, tmp_path):
        cfg = VLASFTRunnerCfg(
            model_type="test_mock_sft",
            dataset_cfg=_MockDatasetCfg(),
            kl_coef=0.01,
            batch_size=16,
            num_epochs=2,
            num_workers=0,
            log_interval=5,
            save_interval=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLASFTRunner(cfg, log_dir=str(tmp_path / "log"), device="cpu")
        assert runner.ref_policy is not None
        runner.learn(num_epochs=2)

    def test_sft_freeze_backbone(self, tmp_path):
        cfg = VLASFTRunnerCfg(
            model_type="test_mock_sft",
            dataset_cfg=_MockDatasetCfg(),
            freeze_backbone=True,
            batch_size=16,
            num_epochs=1,
            num_workers=0,
            save_interval=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLASFTRunner(cfg, log_dir=str(tmp_path / "log"), device="cpu")

        # Backbone params should be frozen
        for p in runner.policy.backbone.parameters():
            assert not p.requires_grad
        # Action head should still be trainable
        for p in runner.policy.action_head.parameters():
            assert p.requires_grad

        runner.learn(num_epochs=1)

    def test_sft_checkpoint_roundtrip(self, tmp_path):
        cfg = VLASFTRunnerCfg(
            model_type="test_mock_sft",
            dataset_cfg=_MockDatasetCfg(),
            batch_size=16,
            num_epochs=2,
            num_workers=0,
            save_interval=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLASFTRunner(cfg, log_dir=str(tmp_path / "log"), device="cpu")
        runner.learn(num_epochs=2)

        # Save
        runner.save_checkpoint(tag="test")
        ckpt_path = tmp_path / "ckpt" / "sft_checkpoint_test.pt"
        assert ckpt_path.exists()

        # Load into new runner
        runner2 = VLASFTRunner(cfg, log_dir=str(tmp_path / "log2"), device="cpu")
        runner2.load_checkpoint(str(ckpt_path))
        assert runner2.global_step == runner.global_step

        # Weights should match
        for (n1, p1), (n2, p2) in zip(
            runner.policy.state_dict().items(),
            runner2.policy.state_dict().items(),
        ):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    def test_sft_cosine_scheduler(self, tmp_path):
        cfg = VLASFTRunnerCfg(
            model_type="test_mock_sft",
            dataset_cfg=_MockDatasetCfg(),
            scheduler_type="cosine",
            batch_size=16,
            num_epochs=2,
            num_workers=0,
            save_interval=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLASFTRunner(cfg, log_dir=str(tmp_path / "log"), device="cpu")
        runner.learn(num_epochs=2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_sft_runner_gpu(self, tmp_path):
        cfg = VLASFTRunnerCfg(
            model_type="test_mock_sft",
            dataset_cfg=_MockDatasetCfg(),
            batch_size=16,
            num_epochs=2,
            num_workers=0,
            save_interval=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        runner = VLASFTRunner(cfg, log_dir=str(tmp_path / "log"), device="cuda")
        runner.learn(num_epochs=2)
