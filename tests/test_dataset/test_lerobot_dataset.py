"""
Tests for LeRobotDataset (v2 format + Psi0 compatibility)

IO tested:
    __init__(cfg) -> dataset with auto-detected format
    __getitem__(idx) -> dict with action, state, proprioception tensors
    __getitem__ with load_videos -> includes "image" (C, H, W) tensor
    get_episode(idx) -> list of frame dicts
    get_stats(key) -> normalization stats dict
    normalize / denormalize -> roundtrip
"""

import pytest
import torch

PSI0_DATA_ROOT = "/vepfs/users/zza/hvla/data/psi-data-shared/unitree_dex3_converted/G1_Dex3_PickApple"


def _psi0_available():
    from pathlib import Path
    return Path(PSI0_DATA_ROOT).exists()


requires_psi0 = pytest.mark.skipif(
    not _psi0_available(), reason="Psi0 data not available"
)

from RoboRenForce.dataset.lerobot.lerobot_dataset import LeRobotDataset, LeRobotDatasetCfg


@requires_psi0
class TestPsi0Dataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        cfg = LeRobotDatasetCfg(data_root=PSI0_DATA_ROOT, load_videos=False)
        return cfg.class_type(cfg)

    def test_metadata(self, dataset):
        assert dataset.action_dim == 36
        assert dataset.proprio_dim == 32
        assert dataset.total_frames == 152569
        assert len(dataset) == 152569

    def test_getitem_keys(self, dataset):
        sample = dataset[0]
        assert "action" in sample
        assert "proprioception" in sample
        assert "observation.state" in sample
        assert "episode_index" in sample
        assert "frame_index" in sample
        assert "timestamp" in sample
        assert "next.done" in sample

    def test_getitem_shapes(self, dataset):
        sample = dataset[0]
        assert sample["action"].shape == (36,)
        assert sample["proprioception"].shape == (32,)
        assert sample["observation.state"].shape == (32,)
        assert sample["action"].dtype == torch.float32

    def test_getitem_boundary(self, dataset):
        first = dataset[0]
        last = dataset[len(dataset) - 1]
        assert first["frame_index"] == 0
        assert isinstance(last["action"], torch.Tensor)

    def test_dataloader(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        assert batch["action"].shape == (16, 36)
        assert batch["proprioception"].shape == (16, 32)

    def test_get_episode(self, dataset):
        ep = dataset.get_episode(0)
        assert len(ep) > 0
        assert all("action" in f for f in ep)
        # Episode 0 has 1179 frames
        assert len(ep) == 1179

    def test_stats(self, dataset):
        stats = dataset.get_stats("action")
        assert "mean" in stats
        assert "std" in stats
        assert stats["mean"].shape == (36,)

    def test_normalize_denormalize_roundtrip(self, dataset):
        sample = dataset[0]
        action = sample["action"]
        normalized = dataset.normalize(action, "action")
        recovered = dataset.denormalize(normalized, "action")
        assert torch.allclose(action, recovered, atol=1e-5)


@requires_psi0
class TestPsi0VideoLoading:
    @pytest.fixture(scope="class")
    def dataset_with_video(self):
        cfg = LeRobotDatasetCfg(
            data_root=PSI0_DATA_ROOT,
            load_videos=True,
            image_size=(224, 224),
        )
        return cfg.class_type(cfg)

    def test_video_frame_shape(self, dataset_with_video):
        sample = dataset_with_video[0]
        assert "image" in sample
        assert sample["image"].shape == (3, 224, 224)
        assert sample["image"].dtype == torch.float32

    def test_video_frame_range(self, dataset_with_video):
        sample = dataset_with_video[0]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_video_custom_size(self):
        cfg = LeRobotDatasetCfg(
            data_root=PSI0_DATA_ROOT,
            load_videos=True,
            image_size=(128, 128),
        )
        ds = cfg.class_type(cfg)
        sample = ds[0]
        assert sample["image"].shape == (3, 128, 128)
