# Tests Structure

**Root**: `tests/`

This document details the test suite organization for VLA integration.

---

## Directory Overview

```
tests/
├── components/         # Component unit tests
├── networks/          # Network unit tests
├── dataset/           # Dataset unit tests
├── algorithms/        # Algorithm unit tests
├── runners/           # Runner integration tests
├── utils/             # Utility tests
└── integration/       # End-to-end integration tests
```

---

## 1. Component Tests: `tests/components/`

### 1.1 Actor Tests

```
tests/components/actor/
├── __init__.py
├── test_vla_actor.py              # VLA actor tests ⭐
└── test_action_heads.py           # Action head tests ⭐
```

#### File: `test_vla_actor.py` (Phase 2, Week 2, Priority P0)

**Purpose**: Unit tests for VLA actor

```python
"""
Unit tests for VLA Actor.

Tests:
- Forward pass
- Frozen VLM gradients
- Config serialization
- Dimension inference
"""

import pytest
import torch

from RoboRenForce.components.actor.vla_actor import VLAActorCfg, VLAActor
from RoboRenForce.networks.vlm.qwen3vl import Qwen3VLCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads import DiffusionActionHeadCfg


@pytest.fixture
def vla_actor_cfg():
    """Create minimal VLA actor config for testing."""
    vlm_cfg = Qwen3VLCfg(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        freeze=True,
        output_dim=2048,
    )
    
    fusion_cfg = FusionLayerCfg(
        output_dim=512,
    )
    
    action_head_cfg = DiffusionActionHeadCfg(
        num_layers=2,
        num_heads=4,
        embed_dim=128,
        num_diffusion_steps=5,
        action_horizon=1,
        action_dim=19,
    )
    
    cfg = VLAActorCfg(
        vlm_backbone_cfg=vlm_cfg,
        freeze_vlm=True,
        fusion_cfg=fusion_cfg,
        action_head_cfg=action_head_cfg,
        use_proprioception=True,
        use_text=False,
    )
    
    return cfg


@pytest.fixture
def dim_params():
    """Dimension parameters for VLA actor."""
    return {
        "proprioception_dim": 48,
        "action_dim": 19,
    }


def test_vla_actor_construction(vla_actor_cfg, dim_params):
    """Test VLA actor can be constructed from config."""
    actor = vla_actor_cfg.construct_from_cfg(dim_params)
    
    assert isinstance(actor, VLAActor)
    assert actor.cfg == vla_actor_cfg


def test_vla_actor_forward(vla_actor_cfg, dim_params):
    """Test VLA actor forward pass."""
    actor = vla_actor_cfg.construct_from_cfg(dim_params)
    actor.eval()
    
    # Create dummy input
    batch_size = 4
    obs_dict = {
        "image": torch.randn(batch_size, 3, 224, 224),
        "proprioception": torch.randn(batch_size, 48),
    }
    
    # Forward pass
    with torch.no_grad():
        action = actor(obs_dict, deterministic=True)
    
    # Check output shape
    assert action.shape == (batch_size, 19)


def test_vla_actor_frozen_vlm(vla_actor_cfg, dim_params):
    """Test that VLM is frozen when freeze_vlm=True."""
    actor = vla_actor_cfg.construct_from_cfg(dim_params)
    
    # Check VLM parameters are frozen
    for param in actor.vlm.parameters():
        assert param.requires_grad == False
    
    # Check fusion and action head are trainable
    for param in actor.fusion.parameters():
        assert param.requires_grad == True
    for param in actor.action_head.parameters():
        assert param.requires_grad == True


def test_vla_actor_config_serialization(vla_actor_cfg):
    """Test config can be serialized and deserialized."""
    # Serialize
    cfg_dict = vla_actor_cfg.to_dict()
    
    # Deserialize
    cfg_restored = VLAActorCfg()
    cfg_restored.from_dict(cfg_dict)
    
    # Check equality
    assert cfg_restored == vla_actor_cfg


def test_vla_actor_with_text(vla_actor_cfg, dim_params):
    """Test VLA actor with text input."""
    vla_actor_cfg.use_text = True
    actor = vla_actor_cfg.construct_from_cfg(dim_params)
    actor.eval()
    
    batch_size = 2
    obs_dict = {
        "image": torch.randn(batch_size, 3, 224, 224),
        "text": torch.randint(0, 1000, (batch_size, 32)),  # Tokenized text
        "proprioception": torch.randn(batch_size, 48),
    }
    
    with torch.no_grad():
        action = actor(obs_dict, deterministic=True)
    
    assert action.shape == (batch_size, 19)


def test_vla_actor_no_proprioception(vla_actor_cfg, dim_params):
    """Test VLA actor without proprioception input."""
    vla_actor_cfg.use_proprioception = False
    dim_params["proprioception_dim"] = 0
    
    actor = vla_actor_cfg.construct_from_cfg(dim_params)
    actor.eval()
    
    batch_size = 2
    obs_dict = {
        "image": torch.randn(batch_size, 3, 224, 224),
    }
    
    with torch.no_grad():
        action = actor(obs_dict, deterministic=True)
    
    assert action.shape == (batch_size, 19)
```

---

#### File: `test_action_heads.py` (Phase 2, Week 2, Priority P0)

**Purpose**: Unit tests for action heads

```python
"""
Unit tests for action heads.

Tests:
- Diffusion action head
- Regression action head
- DDIM sampling
- Training forward
"""

import pytest
import torch

from RoboRenForce.components.actor.action_heads import (
    DiffusionActionHeadCfg,
    DiffusionActionHead,
    RegressionActionHeadCfg,
    RegressionActionHead,
)


def test_diffusion_action_head_construction():
    """Test diffusion action head construction."""
    cfg = DiffusionActionHeadCfg(
        num_layers=2,
        num_heads=4,
        embed_dim=128,
        num_diffusion_steps=10,
        action_horizon=1,
        action_dim=19,
    )
    
    dim_params = {"input_dim": 512}
    
    head = cfg.construct_from_cfg(dim_params)
    assert isinstance(head, DiffusionActionHead)


def test_diffusion_action_head_sampling():
    """Test DDIM sampling during inference."""
    cfg = DiffusionActionHeadCfg(
        num_layers=2,
        num_heads=4,
        embed_dim=128,
        num_diffusion_steps=5,
        action_horizon=1,
        action_dim=19,
    )
    
    head = cfg.construct_from_cfg({"input_dim": 512})
    head.eval()
    
    batch_size = 4
    features = torch.randn(batch_size, 512)
    
    with torch.no_grad():
        action = head(features, deterministic=True)
    
    assert action.shape == (batch_size, 1, 19)


def test_diffusion_action_head_training():
    """Test training forward pass."""
    cfg = DiffusionActionHeadCfg(
        num_layers=2,
        num_heads=4,
        embed_dim=128,
        num_diffusion_steps=10,
        action_horizon=1,
        action_dim=19,
    )
    
    head = cfg.construct_from_cfg({"input_dim": 512})
    head.train()
    
    batch_size = 4
    features = torch.randn(batch_size, 512)
    
    # Training forward (denoising)
    output = head.train_forward(features)
    
    # Output should be noise prediction
    assert "noise_pred" in output or "loss" in output


def test_regression_action_head():
    """Test regression (MLP) action head."""
    cfg = RegressionActionHeadCfg(
        hidden_dims=[256, 256],
        activation="relu",
        action_dim=19,
    )
    
    head = cfg.construct_from_cfg({"input_dim": 512})
    
    batch_size = 4
    features = torch.randn(batch_size, 512)
    
    action = head(features, deterministic=True)
    
    assert action.shape == (batch_size, 19)
```

---

## 2. Network Tests: `tests/networks/`

### 2.1 VLM Tests

```
tests/networks/vlm/
├── __init__.py
├── test_vlm_backbone.py           # VLM backbone tests ⭐
├── test_qwen3vl.py               # Qwen3-VL tests ⭐
└── test_fusion_layers.py         # Fusion layer tests ⭐
```

#### File: `test_qwen3vl.py` (Phase 1, Week 1, Priority P0)

**Purpose**: Unit tests for Qwen3-VL wrapper

```python
"""
Unit tests for Qwen3-VL wrapper.

Tests:
- Model loading
- Feature extraction
- Freezing
- LoRA application
"""

import pytest
import torch

from RoboRenForce.networks.vlm.qwen3vl import Qwen3VLCfg, Qwen3VL


@pytest.mark.slow
def test_qwen3vl_loading():
    """Test Qwen3-VL can be loaded from HuggingFace."""
    cfg = Qwen3VLCfg(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        freeze=True,
        output_dim=2048,
    )
    
    model = cfg.construct_from_cfg()
    
    assert isinstance(model, Qwen3VL)


@pytest.mark.slow
def test_qwen3vl_forward():
    """Test Qwen3-VL forward pass."""
    cfg = Qwen3VLCfg(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        freeze=True,
        output_dim=2048,
    )
    
    model = cfg.construct_from_cfg()
    model.eval()
    
    # Dummy input
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        features = model(image, text=None)
    
    assert features.shape == (batch_size, 2048)


def test_qwen3vl_frozen():
    """Test that freeze=True freezes all parameters."""
    cfg = Qwen3VLCfg(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        freeze=True,
    )
    
    model = cfg.construct_from_cfg()
    
    for param in model.model.parameters():
        assert param.requires_grad == False


def test_qwen3vl_lora():
    """Test LoRA application."""
    cfg = Qwen3VLCfg(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        freeze=False,
        use_lora=True,
        lora_rank=8,
    )
    
    model = cfg.construct_from_cfg()
    
    # Check that some parameters are trainable (LoRA)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0
```

---

## 3. Dataset Tests: `tests/dataset/`

```
tests/dataset/
├── __init__.py
├── test_lerobot_dataset.py       # LeRobot dataset tests ⭐
└── test_lerobot_processor.py     # Processor tests ⭐
```

#### File: `test_lerobot_dataset.py` (Phase 1, Week 1, Priority P0)

**Purpose**: Unit tests for LeRobot dataset

```python
"""
Unit tests for LeRobot dataset.

Tests:
- Dataset loading
- Episode iteration
- Data format validation
- Processor integration
"""

import pytest
import torch
from pathlib import Path

from RoboRenForce.dataset.lerobot import LeRobotDatasetCfg, LeRobotDataset


@pytest.fixture
def test_dataset_path(tmp_path):
    """Create a minimal test dataset."""
    # Create directory structure
    data_root = tmp_path / "test_dataset"
    (data_root / "meta").mkdir(parents=True)
    (data_root / "train").mkdir()
    
    # Create dummy metadata
    # (In real test, create minimal Parquet + stats.safetensors)
    
    return data_root


def test_lerobot_dataset_construction(test_dataset_path):
    """Test dataset can be constructed."""
    cfg = LeRobotDatasetCfg(
        data_root=str(test_dataset_path),
        split="train",
        load_videos=False,
    )
    
    dataset = cfg.construct_from_cfg()
    assert isinstance(dataset, LeRobotDataset)


def test_lerobot_dataset_getitem(test_dataset_path):
    """Test __getitem__ returns correct format."""
    cfg = LeRobotDatasetCfg(
        data_root=str(test_dataset_path),
        split="train",
        load_videos=False,
    )
    
    dataset = cfg.construct_from_cfg()
    
    # Get first sample
    sample = dataset[0]
    
    # Check keys
    assert "image" in sample
    assert "action" in sample
    assert "proprioception" in sample or "state" in sample
    
    # Check types
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["action"], torch.Tensor)
```

---

## 4. Algorithm Tests: `tests/algorithms/`

```
tests/algorithms/vla_training/
├── __init__.py
├── test_pretrain_algorithm.py    # Pretrain algorithm tests ⭐
├── test_sft_algorithm.py         # SFT algorithm tests
└── test_rl_finetune_algorithm.py # RL finetune tests
```

#### File: `test_pretrain_algorithm.py` (Phase 3, Week 3, Priority P1)

**Purpose**: Unit tests for VLA pretraining algorithm

```python
"""
Unit tests for VLA pretrain algorithm.

Tests:
- Loss computation
- Optimizer update
- Mixed precision
- Gradient clipping
"""

import pytest
import torch

from RoboRenForce.algorithms.vla_training.pretrain_algorithm import (
    VLAPretrainAlgorithmCfg,
    VLAPretrainAlgorithm,
)


def test_pretrain_algorithm_construction():
    """Test algorithm construction."""
    cfg = VLAPretrainAlgorithmCfg(
        action_loss_weight=1.0,
        learning_rate=1e-4,
        use_amp=False,
    )
    
    algorithm = cfg.construct_from_cfg()
    assert isinstance(algorithm, VLAPretrainAlgorithm)


def test_pretrain_algorithm_loss_computation():
    """Test loss computation."""
    cfg = VLAPretrainAlgorithmCfg(
        action_loss_weight=1.0,
        use_amp=False,
    )
    
    algorithm = cfg.construct_from_cfg()
    
    # Dummy batch
    batch = {
        "action": torch.randn(4, 19),
    }
    
    # Dummy VLA actor (mock)
    class MockVLAActor:
        def __call__(self, batch):
            return torch.randn(4, 19)  # Predicted action
    
    vla_actor = MockVLAActor()
    
    # Compute loss
    loss_dict = algorithm.compute_loss(batch, vla_actor)
    
    assert "total_loss" in loss_dict
    assert "action_loss" in loss_dict
    assert loss_dict["total_loss"].item() >= 0
```

---

## 5. Runner Tests: `tests/runners/`

```
tests/runners/vla/
├── __init__.py
├── pretrain/
│   ├── __init__.py
│   └── test_vla_pretrain_runner.py      # Pretrain runner tests ⭐
├── post_train/
│   ├── __init__.py
│   └── test_vla_sft_runner.py
└── rl/
    ├── __init__.py
    └── test_vla_rl_runner.py
```

#### File: `test_vla_pretrain_runner.py` (Phase 3, Week 3, Priority P1)

**Purpose**: Integration test for VLA pretraining runner

```python
"""
Integration tests for VLA pretrain runner.

Tests:
- Runner construction
- Training loop (short)
- Checkpoint saving/loading
- Logging
"""

import pytest
import torch
from pathlib import Path

from RoboRenForce.runners.vla.pretrain import VLAPretrainRunnerCfg, VLAPretrainRunner


@pytest.fixture
def minimal_runner_cfg(test_dataset_path, tmp_path):
    """Create minimal runner config for testing."""
    # Use configs from earlier tests
    # ... (similar to test config in RRF_vla_tasks)
    pass


@pytest.mark.integration
def test_pretrain_runner_construction(minimal_runner_cfg, tmp_path):
    """Test runner can be constructed."""
    runner = VLAPretrainRunner(
        cfg=minimal_runner_cfg,
        log_dir=str(tmp_path / "logs"),
        device="cpu",  # Use CPU for testing
    )
    
    assert isinstance(runner, VLAPretrainRunner)


@pytest.mark.integration
def test_pretrain_runner_short_training(minimal_runner_cfg, tmp_path):
    """Test short training run (10 iterations)."""
    runner = VLAPretrainRunner(
        cfg=minimal_runner_cfg,
        log_dir=str(tmp_path / "logs"),
        device="cpu",
    )
    
    # Train for 1 epoch (small dataset)
    runner.learn(num_epochs=1)
    
    # Check that training happened
    assert runner.global_step > 0


@pytest.mark.integration
def test_pretrain_runner_checkpoint(minimal_runner_cfg, tmp_path):
    """Test checkpoint saving and loading."""
    cfg = minimal_runner_cfg
    cfg.checkpoint_dir = str(tmp_path / "checkpoints")
    
    runner = VLAPretrainRunner(
        cfg=cfg,
        log_dir=str(tmp_path / "logs"),
        device="cpu",
    )
    
    # Train a bit
    runner.learn(num_epochs=1)
    
    # Save checkpoint
    runner.save_checkpoint()
    
    # Check checkpoint exists
    ckpt_files = list(Path(cfg.checkpoint_dir).glob("*.pth"))
    assert len(ckpt_files) > 0
    
    # Load checkpoint in new runner
    runner2 = VLAPretrainRunner(
        cfg=cfg,
        log_dir=str(tmp_path / "logs2"),
        device="cpu",
    )
    runner2.load_checkpoint(ckpt_files[0])
    
    assert runner2.global_step == runner.global_step
```

---

## 6. Integration Tests: `tests/integration/`

```
tests/integration/
├── __init__.py
├── test_end_to_end_pretrain.py   # Full pretrain pipeline ⭐
├── test_end_to_end_sft.py        # Pretrain → SFT pipeline
└── test_end_to_end_rl.py         # SFT → RL pipeline
```

#### File: `test_end_to_end_pretrain.py` (Phase 4, Week 4, Priority P1)

**Purpose**: End-to-end test of full VLA pretraining pipeline

```python
"""
End-to-end integration test for VLA pretraining.

Tests:
- Data loading → Training → Checkpoint saving
- Config → Runner → VLA Actor pipeline
- Logging and metrics
"""

import pytest
from pathlib import Path


@pytest.mark.slow
@pytest.mark.integration
def test_full_pretrain_pipeline(tmp_path):
    """
    Full VLA pretrain pipeline test.
    
    Steps:
    1. Load config
    2. Create dataset
    3. Create VLA actor
    4. Train for a few steps
    5. Save checkpoint
    6. Validate checkpoint
    """
    # Implementation
    pass
```

---

## 7. Utility Tests: `tests/utils/`

```
tests/utils/vla/
├── __init__.py
├── test_distributed_utils.py     # DDP utils tests
├── test_processor.py             # Image/text processor tests
└── test_video_utils.py           # Video I/O tests
```

---

## Testing Strategy

### 7.1 Test Levels

1. **Unit Tests** (Fast, isolated)
   - Components, networks, algorithms
   - Run on every commit
   - No GPU required (use CPU or mock)

2. **Integration Tests** (Medium speed)
   - Runners, dataset loaders
   - Run before PR merge
   - May require small GPU

3. **End-to-End Tests** (Slow)
   - Full training pipelines
   - Run nightly or on release
   - Require GPU + datasets

### 7.2 Test Markers

```python
# Mark slow tests (skip in CI)
@pytest.mark.slow

# Mark integration tests
@pytest.mark.integration

# Mark GPU-required tests
@pytest.mark.gpu

# Mark tests requiring datasets
@pytest.mark.dataset
```

### 7.3 Running Tests

```bash
# Run all tests
pytest tests/

# Run fast tests only
pytest tests/ -m "not slow"

# Run specific test file
pytest tests/components/actor/test_vla_actor.py

# Run with coverage
pytest tests/ --cov=RoboRenForce --cov-report=html
```

---

## Priority Summary

### Phase 1 (Week 1): **3 files** (P0)
- ✅ `tests/networks/vlm/test_qwen3vl.py`
- ✅ `tests/dataset/test_lerobot_dataset.py`
- ✅ `tests/dataset/test_lerobot_processor.py`

### Phase 2 (Week 2): **2 files** (P0)
- ✅ `tests/components/actor/test_vla_actor.py`
- ✅ `tests/components/actor/test_action_heads.py`

### Phase 3 (Week 3): **2 files** (P1)
- ✅ `tests/algorithms/vla_training/test_pretrain_algorithm.py`
- ✅ `tests/runners/vla/pretrain/test_vla_pretrain_runner.py`

### Phase 4 (Week 4): **2 files** (P1)
- ✅ `tests/utils/vla/test_distributed_utils.py`
- ✅ `tests/integration/test_end_to_end_pretrain.py`

### Phase 5-6 (Weeks 5-6): **2 files** (P2)
- `tests/runners/vla/post_train/test_vla_sft_runner.py`
- `tests/runners/vla/rl/test_vla_rl_runner.py`

### Phase 7+ (Future): **3 files** (P2)
- `tests/integration/test_end_to_end_sft.py`
- `tests/integration/test_end_to_end_rl.py`
- `tests/utils/vla/test_video_utils.py`

---

## Test Directory Structure Summary

```
tests/
├── __init__.py
├── components/
│   ├── __init__.py
│   └── actor/
│       ├── __init__.py
│       ├── test_vla_actor.py         ⭐ Phase 2
│       └── test_action_heads.py      ⭐ Phase 2
├── networks/
│   ├── __init__.py
│   └── vlm/
│       ├── __init__.py
│       ├── test_vlm_backbone.py
│       ├── test_qwen3vl.py          ⭐ Phase 1
│       └── test_fusion_layers.py
├── dataset/
│   ├── __init__.py
│   ├── test_lerobot_dataset.py      ⭐ Phase 1
│   └── test_lerobot_processor.py    ⭐ Phase 1
├── algorithms/
│   ├── __init__.py
│   └── vla_training/
│       ├── __init__.py
│       ├── test_pretrain_algorithm.py  ⭐ Phase 3
│       ├── test_sft_algorithm.py
│       └── test_rl_finetune_algorithm.py
├── runners/
│   ├── __init__.py
│   └── vla/
│       ├── __init__.py
│       ├── pretrain/
│       │   ├── __init__.py
│       │   └── test_vla_pretrain_runner.py  ⭐ Phase 3
│       ├── post_train/
│       │   ├── __init__.py
│       │   └── test_vla_sft_runner.py
│       └── rl/
│           ├── __init__.py
│           └── test_vla_rl_runner.py
├── utils/
│   ├── __init__.py
│   └── vla/
│       ├── __init__.py
│       ├── test_distributed_utils.py  ⭐ Phase 4
│       ├── test_processor.py
│       └── test_video_utils.py
└── integration/
    ├── __init__.py
    ├── test_end_to_end_pretrain.py    ⭐ Phase 4
    ├── test_end_to_end_sft.py
    └── test_end_to_end_rl.py
```

**Total Test Files**: ~25 files (including __init__.py)

**Core Tests (P0-P1)**: 11 files
- Phase 1: 3 files (dataset, VLM)
- Phase 2: 2 files (actor, action heads)
- Phase 3: 2 files (algorithm, runner)
- Phase 4: 2 files (distributed, integration)

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: VLA Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: |
          pytest tests/ -m "not slow and not gpu"
  
  integration-tests:
    runs-on: ubuntu-gpu
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: |
          pytest tests/ -m "integration and not slow"
```

---

## Best Practices

1. **Fast Unit Tests**: Use CPU, mock external dependencies
2. **Fixtures**: Reuse test configs and datasets via pytest fixtures
3. **Markers**: Tag tests appropriately (slow, gpu, integration)
4. **Coverage**: Aim for >80% coverage on core components
5. **CI**: Run fast tests on every commit, slow tests nightly
