# VLA Tasks Structure

**Root**: `source/RRF_vla_tasks/RRF_vla_tasks/`

This document details the VLA task configuration files (example configs for VLA training).

---

## Directory Overview

```
RRF_vla_tasks/
├── vla_pretrain/       # Pretraining configs
├── vla_post_train/     # SFT/DPO configs
└── vla_rl_finetune/    # RL fine-tuning configs
```

**Purpose**: Provide example configurations for each VLA training stage, similar to how `RRF_isaaclab_tasks` provides RL task configs.

---

## 1. Pretraining Configs: `vla_pretrain/`

**Phase 3-4** (Weeks 3-4, Priority P0)

```
vla_pretrain/
├── __init__.py
├── minimal_example.py              # Minimal config for quick testing ⭐
├── humanoid_qwen3vl.py            # Full humanoid config with Qwen3-VL ⭐
├── single_robot_baseline.py       # Single robot, simple action head
└── multi_robot_mixture.py         # Mixed robot data (Future)
```

### File: `minimal_example.py`

**Purpose**: Quick sanity check for single-GPU training (Phase 3)

```python
"""
Minimal VLA pretraining config for testing.

Usage:
    python scripts/vla/pretrain/train_single_gpu.py \\
        --config RRF_vla_tasks.vla_pretrain.minimal_example
"""

from RoboRenForce.runners.vla.pretrain import VLAPretrainRunnerCfg
from RoboRenForce.components.actor.vla_actor import VLAActorCfg
from RoboRenForce.networks.vlm.qwen3vl import Qwen3VLCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads import RegressionActionHeadCfg
from RoboRenForce.dataset.lerobot import LeRobotDatasetCfg, LeRobotProcessorCfg
from RoboRenForce.algorithms.vla_training.pretrain_algorithm import VLAPretrainAlgorithmCfg

def get_config():
    """Minimal config: small dataset, simple MLP action head, 1 GPU."""
    
    # Dataset: Small test dataset
    dataset_cfg = LeRobotDatasetCfg(
        data_root="data/test_humanoid_demos",  # ~100 episodes
        split="train",
        processor_cfg=LeRobotProcessorCfg(
            image_size=(224, 224),
            normalize_actions=True,
        ),
        load_videos=True,
        num_workers=4,
    )
    
    # VLM Backbone: Qwen3-VL-2B (frozen)
    vlm_cfg = Qwen3VLCfg(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        freeze=True,
        output_dim=2048,
    )
    
    # Fusion Layer
    fusion_cfg = FusionLayerCfg(
        fusion_type="concat_mlp",
        output_dim=512,
        hidden_dims=[512],
    )
    
    # Action Head: Simple MLP (not diffusion, for speed)
    action_head_cfg = RegressionActionHeadCfg(
        hidden_dims=[256, 256],
        activation="relu",
        action_dim=19,  # Humanoid action dim
    )
    
    # VLA Actor
    vla_actor_cfg = VLAActorCfg(
        vlm_backbone_cfg=vlm_cfg,
        freeze_vlm=True,
        fusion_cfg=fusion_cfg,
        action_head_cfg=action_head_cfg,
        use_proprioception=True,
        use_text=False,  # No text for minimal test
    )
    
    # Algorithm
    algorithm_cfg = VLAPretrainAlgorithmCfg(
        action_loss_weight=1.0,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        max_grad_norm=1.0,
        use_amp=True,
        amp_dtype="bf16",
    )
    
    # Runner
    runner_cfg = VLAPretrainRunnerCfg(
        vla_actor_cfg=vla_actor_cfg,
        algorithm_cfg=algorithm_cfg,
        dataset_cfg=dataset_cfg,
        batch_size=16,  # Small batch for testing
        num_epochs=2,
        num_workers=4,
        save_interval=500,
        checkpoint_dir="checkpoints/minimal_test",
    )
    
    return runner_cfg
```

### File: `humanoid_qwen3vl.py`

**Purpose**: Full production config for humanoid VLA pretraining (Phase 4, DDP)

```python
"""
Full humanoid VLA pretraining config with Qwen3-VL.

Usage (8 GPUs):
    torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \\
        --config RRF_vla_tasks.vla_pretrain.humanoid_qwen3vl
"""

from RoboRenForce.runners.vla.pretrain import DistributedVLAPretrainRunnerCfg
from RoboRenForce.components.actor.action_heads import DiffusionActionHeadCfg

def get_config():
    """Production config: large dataset, diffusion action head, 8 GPUs."""
    
    # Dataset: Full humanoid dataset (~10k episodes)
    dataset_cfg = LeRobotDatasetCfg(
        data_root="data/humanoid_mixed_tasks",
        split="train",
        processor_cfg=LeRobotProcessorCfg(
            image_size=(224, 224),
            normalize_actions=True,
            normalize_proprioception=True,
        ),
        load_videos=True,
        video_backend="pyav",  # Faster than opencv
        num_workers=8,
    )
    
    # VLM Backbone: Qwen3-VL-2B (frozen)
    vlm_cfg = Qwen3VLCfg(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        freeze=True,
        output_dim=2048,
    )
    
    # Fusion Layer
    fusion_cfg = FusionLayerCfg(
        fusion_type="concat_mlp",
        output_dim=512,
        hidden_dims=[512, 512],
    )
    
    # Action Head: Diffusion Transformer ⭐
    action_head_cfg = DiffusionActionHeadCfg(
        num_layers=4,
        num_heads=8,
        embed_dim=256,
        num_diffusion_steps=10,
        noise_schedule="cosine",
        action_horizon=1,
        action_dim=19,
    )
    
    # VLA Actor
    vla_actor_cfg = VLAActorCfg(
        vlm_backbone_cfg=vlm_cfg,
        freeze_vlm=True,
        fusion_cfg=fusion_cfg,
        action_head_cfg=action_head_cfg,
        use_proprioception=True,
        use_text=True,  # Use text instructions
    )
    
    # Algorithm
    algorithm_cfg = VLAPretrainAlgorithmCfg(
        action_loss_weight=1.0,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        max_grad_norm=1.0,
        use_amp=True,
        amp_dtype="bf16",
    )
    
    # Distributed Runner (8 GPUs)
    runner_cfg = DistributedVLAPretrainRunnerCfg(
        vla_actor_cfg=vla_actor_cfg,
        algorithm_cfg=algorithm_cfg,
        dataset_cfg=dataset_cfg,
        batch_size=8,  # Per GPU (total: 8*8=64)
        num_epochs=20,
        num_workers=8,
        save_interval=1000,
        checkpoint_dir="checkpoints/humanoid_qwen3vl",
        backend="nccl",
        find_unused_parameters=False,
    )
    
    return runner_cfg
```

### File: `single_robot_baseline.py`

**Purpose**: Baseline config for ablation studies

```python
"""
Baseline: Single robot, MLP action head, no diffusion.

For ablation studies comparing action head architectures.
"""

def get_config():
    """Baseline config with simplest components."""
    
    # Similar to minimal_example but:
    # - Full dataset (not test)
    # - Single robot type only
    # - MLP action head (for comparison with Diffusion)
    
    dataset_cfg = LeRobotDatasetCfg(
        data_root="data/humanoid_single_task",  # One task only
        split="train",
        ...
    )
    
    action_head_cfg = RegressionActionHeadCfg(
        hidden_dims=[512, 512, 256],
        activation="relu",
        action_dim=19,
    )
    
    # ... rest similar to humanoid_qwen3vl
    
    return runner_cfg
```

### File: `multi_robot_mixture.py`

**Purpose**: Mixed robot data training (Future, Phase 7+)

```python
"""
Multi-robot mixture training.

Trains VLA on data from multiple robot types:
- Humanoid
- Quadruped
- Manipulator

Uses mixture dataset sampler.
"""

from RoboRenForce.dataset.mixture import MixtureDatasetCfg

def get_config():
    """Multi-robot mixture config."""
    
    # Mixture dataset
    dataset_cfg = MixtureDatasetCfg(
        datasets=[
            LeRobotDatasetCfg(data_root="data/humanoid_demos", ...),
            LeRobotDatasetCfg(data_root="data/quadruped_demos", ...),
            LeRobotDatasetCfg(data_root="data/manipulator_demos", ...),
        ],
        sampling_weights=[0.5, 0.3, 0.2],  # Humanoid-heavy
    )
    
    # ... rest similar
    
    return runner_cfg
```

---

## 2. Post-Training Configs: `vla_post_train/`

**Phase 5** (Week 5, Priority P1)

```
vla_post_train/
├── __init__.py
├── sft_humanoid_reach.py         # SFT for reach task ⭐
├── sft_humanoid_walk.py          # SFT for locomotion
└── dpo_humanoid.py               # DPO (Future)
```

### File: `sft_humanoid_reach.py`

**Purpose**: Task-specific SFT on pretrained VLA

```python
"""
SFT config for humanoid reach task.

Usage (4 GPUs):
    torchrun --nproc_per_node=4 scripts/vla/post_train/train_sft.py \\
        --config RRF_vla_tasks.vla_post_train.sft_humanoid_reach \\
        --pretrained checkpoints/humanoid_qwen3vl/vla_step_10000.pth
"""

from RoboRenForce.runners.vla.post_train import VLASFTRunnerCfg
from RoboRenForce.algorithms.vla_training.sft_algorithm import VLASFTAlgorithmCfg

def get_config():
    """SFT config: task-specific fine-tuning."""
    
    # Task-specific dataset
    dataset_cfg = LeRobotDatasetCfg(
        data_root="data/humanoid_reach_expert_demos",  # High-quality demos
        split="train",
        ...
    )
    
    # SFT Algorithm (lower LR than pretraining)
    algorithm_cfg = VLASFTAlgorithmCfg(
        action_loss_weight=1.0,
        learning_rate=5e-5,  # 10x lower than pretrain
        weight_decay=0.01,
        warmup_steps=500,
        max_grad_norm=1.0,
        use_amp=True,
    )
    
    # SFT Runner
    runner_cfg = VLASFTRunnerCfg(
        pretrained_vla_path="checkpoints/humanoid_qwen3vl/vla_step_10000.pth",
        vla_actor_cfg=vla_actor_cfg,  # Same architecture
        algorithm_cfg=algorithm_cfg,
        dataset_cfg=dataset_cfg,
        batch_size=16,  # Per GPU
        num_epochs=5,  # Fewer epochs than pretrain
        save_interval=500,
        checkpoint_dir="checkpoints/sft_reach",
    )
    
    return runner_cfg
```

---

## 3. RL Fine-tuning Configs: `vla_rl_finetune/`

**Phase 6** (Week 6, Priority P1)

```
vla_rl_finetune/
├── __init__.py
├── ppo_humanoid_reach.py         # PPO fine-tuning ⭐
├── ppo_humanoid_walk.py          # PPO for locomotion
└── sac_humanoid_manipulation.py  # SAC fine-tuning
```

### File: `ppo_humanoid_reach.py`

**Purpose**: RL fine-tuning of VLA with PPO

```python
"""
PPO RL fine-tuning config for humanoid reach.

Usage (single GPU, 4096 envs):
    python scripts/vla/rl/train_ppo_finetune.py \\
        --config RRF_vla_tasks.vla_rl_finetune.ppo_humanoid_reach \\
        --pretrained checkpoints/sft_reach/vla_step_5000.pth
"""

from RoboRenForce.runners.vla.rl import VLARLRunnerCfg
from RoboRenForce.algorithms.vla_training.rl_finetune_algorithm import VLARLFinetuneAlgorithmCfg
from RoboRenForce.algorithms.on_policy.ppo import PPOCfg

def get_config():
    """RL fine-tuning config with PPO."""
    
    # Base PPO algorithm
    ppo_cfg = PPOCfg(
        value_loss_coef=1.0,
        entropy_coef=0.01,
        clip_param=0.2,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-5,  # Low LR for fine-tuning
        max_grad_norm=1.0,
    )
    
    # VLA RL fine-tuning algorithm
    rl_finetune_algorithm_cfg = VLARLFinetuneAlgorithmCfg(
        base_rl_algorithm_cfg=ppo_cfg,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        freeze_vlm=True,  # Keep VLM frozen
    )
    
    # VLA Actor config (same as pretrained)
    vla_actor_cfg = VLAActorCfg(
        vlm_backbone_cfg=Qwen3VLCfg(...),
        ...
    )
    
    # VLA RL Runner
    runner_cfg = VLARLRunnerCfg(
        pretrained_vla_path="checkpoints/sft_reach/vla_step_5000.pth",
        vla_actor_cfg=vla_actor_cfg,
        rl_algorithm_cfg=rl_finetune_algorithm_cfg,
        task="Isaac-Humanoid-Reach-v0",
        num_envs=4096,  # Environment parallelism
        max_iterations=1000,
        save_interval=100,
        checkpoint_dir="checkpoints/rl_finetune_reach",
    )
    
    return runner_cfg
```

---

## Directory Structure Summary

```
RRF_vla_tasks/
├── __init__.py
├── vla_pretrain/                 # 4 files (Phase 3-4)
│   ├── __init__.py
│   ├── minimal_example.py       ⭐ Week 3
│   ├── humanoid_qwen3vl.py     ⭐ Week 4
│   ├── single_robot_baseline.py
│   └── multi_robot_mixture.py   # Future
├── vla_post_train/               # 3 files (Phase 5)
│   ├── __init__.py
│   ├── sft_humanoid_reach.py   ⭐ Week 5
│   ├── sft_humanoid_walk.py
│   └── dpo_humanoid.py          # Future
└── vla_rl_finetune/              # 3 files (Phase 6)
    ├── __init__.py
    ├── ppo_humanoid_reach.py   ⭐ Week 6
    ├── ppo_humanoid_walk.py
    └── sac_humanoid_manipulation.py
```

**Total: 11 files** (4 + 3 + 3 + 1 root __init__.py)

---

## Usage Examples

### Phase 3: Single-GPU Testing
```bash
python scripts/vla/pretrain/train_single_gpu.py \\
    --config RRF_vla_tasks.vla_pretrain.minimal_example \\
    --log_dir logs/minimal_test
```

### Phase 4: Multi-GPU DDP Pretraining
```bash
torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \\
    --config RRF_vla_tasks.vla_pretrain.humanoid_qwen3vl \\
    --log_dir logs/pretrain_ddp
```

### Phase 5: SFT Post-Training
```bash
torchrun --nproc_per_node=4 scripts/vla/post_train/train_sft.py \\
    --config RRF_vla_tasks.vla_post_train.sft_humanoid_reach \\
    --pretrained checkpoints/humanoid_qwen3vl/vla_step_10000.pth \\
    --log_dir logs/sft_reach
```

### Phase 6: RL Fine-tuning
```bash
python scripts/vla/rl/train_ppo_finetune.py \\
    --config RRF_vla_tasks.vla_rl_finetune.ppo_humanoid_reach \\
    --pretrained checkpoints/sft_reach/vla_step_5000.pth \\
    --log_dir logs/rl_finetune_reach
```

---

## Priority Breakdown

### Phase 3 (Week 3): **2 files** (P0)
- ✅ `vla_pretrain/minimal_example.py`
- ✅ `vla_pretrain/__init__.py`

### Phase 4 (Week 4): **1 file** (P0)
- ✅ `vla_pretrain/humanoid_qwen3vl.py`

### Phase 5 (Week 5): **2 files** (P1)
- ✅ `vla_post_train/sft_humanoid_reach.py`
- ✅ `vla_post_train/__init__.py`

### Phase 6 (Week 6): **2 files** (P1)
- ✅ `vla_rl_finetune/ppo_humanoid_reach.py`
- ✅ `vla_rl_finetune/__init__.py`

### Phase 7+ (Future): **4 files** (P2)
- `vla_pretrain/single_robot_baseline.py`
- `vla_pretrain/multi_robot_mixture.py`
- `vla_post_train/dpo_humanoid.py`
- `vla_rl_finetune/sac_humanoid_manipulation.py`

---

## Design Principles

1. **Separation from RL Tasks**: VLA configs are in `RRF_vla_tasks`, RL configs stay in `RRF_isaaclab_tasks`

2. **Progressive Complexity**: 
   - Minimal → Full (pretrain)
   - Pretrain → SFT → RL (training stages)

3. **Reusable Components**: 
   - Same VLA actor architecture across stages
   - Only change dataset + algorithm + runner

4. **Example-Driven**: 
   - Each file is a complete, runnable example
   - Users can copy and modify for their tasks

5. **Reference Documentation**:
   - Each config file includes usage examples in docstrings
   - Clear comments explaining hyperparameter choices
