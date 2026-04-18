"""
Tests for LeRobotProcessor

IO tested:
    __call__(raw_data, training) -> processed_data
        - Image (H,W,C) numpy uint8 -> (C,224,224) float tensor normalized
        - State (D,) numpy -> (D,) float tensor normalized
        - Action (A,) numpy -> (A,) float tensor normalized

    normalize(tensor, key) -> normalized tensor
    denormalize(tensor, key) -> original tensor (round-trip)

    process_image(image) -> (C,H',W') float tensor
"""

import numpy as np
import torch
import pytest

from RoboRenForce.dataset.lerobot.lerobot_processor import (
    LeRobotProcessor,
    LeRobotProcessorCfg,
)


@pytest.fixture
def stats():
    return {
        "action": {
            "mean": np.array([0.5, -0.3]),
            "std": np.array([1.0, 2.0]),
        },
        "observation.state": {
            "mean": np.array([1.0, 2.0, 3.0]),
            "std": np.array([0.5, 0.5, 0.5]),
        },
    }


@pytest.fixture
def processor(stats):
    cfg = LeRobotProcessorCfg()
    return LeRobotProcessor(cfg, stats=stats)


@pytest.fixture
def raw_sample():
    return {
        "observation.image.top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "observation.state": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "action": np.array([0.8, -0.1], dtype=np.float32),
        "episode_index": 0,
        "frame_index": 5,
    }


class TestLeRobotProcessorInit:
    def test_create_default(self):
        cfg = LeRobotProcessorCfg()
        proc = LeRobotProcessor(cfg)
        assert proc is not None

    def test_create_with_stats(self, stats):
        cfg = LeRobotProcessorCfg()
        proc = LeRobotProcessor(cfg, stats=stats)
        assert proc.stats == stats

    def test_create_with_augmentation(self):
        cfg = LeRobotProcessorCfg(augment_images=True)
        proc = LeRobotProcessor(cfg)
        assert proc.image_transform is not None


class TestProcessImage:
    def test_numpy_uint8(self, processor):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = processor.process_image(img)
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

    def test_numpy_float(self, processor):
        img = np.random.rand(480, 640, 3).astype(np.float32)
        result = processor.process_image(img)
        assert result.shape == (3, 224, 224)

    def test_tensor_chw(self, processor):
        img = torch.rand(3, 480, 640)
        result = processor.process_image(img)
        assert result.shape == (3, 224, 224)


class TestNormalization:
    def test_normalize(self, processor):
        data = torch.tensor([0.5, -0.3])
        normed = processor.normalize(data, "action")
        # (0.5 - 0.5) / (1.0 + 1e-8) ≈ 0, (-0.3 - -0.3) / (2.0 + 1e-8) ≈ 0
        assert torch.allclose(normed, torch.zeros(2), atol=1e-6)

    def test_denormalize(self, processor):
        data = torch.tensor([0.0, 0.0])
        denormed = processor.denormalize(data, "action")
        expected = torch.tensor([0.5, -0.3])
        assert torch.allclose(denormed, expected, atol=1e-6)

    def test_round_trip(self, processor):
        original = torch.tensor([1.5, 2.5, 3.5])
        normed = processor.normalize(original, "observation.state")
        recovered = processor.denormalize(normed, "observation.state")
        assert torch.allclose(recovered, original, atol=1e-5)

    def test_unknown_key_passthrough(self, processor):
        data = torch.tensor([1.0, 2.0])
        result = processor.normalize(data, "unknown_key")
        assert torch.equal(result, data)


class TestCall:
    def test_full_sample(self, processor, raw_sample):
        result = processor(raw_sample)
        # Image processed
        assert result["observation.image.top"].shape == (3, 224, 224)
        assert result["observation.image.top"].dtype == torch.float32
        # State normalized
        assert result["observation.state"].shape == (3,)
        # Action normalized
        assert result["action"].shape == (2,)
        # Passthrough scalars
        assert result["episode_index"] == 0

    def test_no_augment_in_eval(self, raw_sample, stats):
        cfg = LeRobotProcessorCfg(augment_images=True)
        proc = LeRobotProcessor(cfg, stats=stats)
        result = proc(raw_sample, training=False)
        assert result["observation.image.top"].shape == (3, 224, 224)
