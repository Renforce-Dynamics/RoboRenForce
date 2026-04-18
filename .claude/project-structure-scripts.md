# Scripts Structure

**Root**: `scripts/`

This document details all training, evaluation, and utility scripts.

---

## Directory Overview

```
scripts/
├── renforce/           # Low-level RL scripts (existing, unchanged)
├── vla/               # VLA training scripts (new) ⭐
├── distributed/       # Distributed training launchers (new) ⭐
├── data/              # Data processing scripts (existing + new)
└── third_party/       # Third-party scripts (existing)
```

---

## 1. Low-Level RL Scripts: `renforce/`

**Current Structure** (Unchanged)

```
renforce/
├── train_lab.py       # Isaac Lab RL training (PPO, SAC, etc.)
├── train_gym.py       # Gym environment RL training
└── play_lab.py        # Evaluate trained RL policies
```

These scripts are for **System 0 (Locomotion)** training and remain unchanged.

### Example Usage
```bash
# Train PPO on humanoid locomotion
python scripts/renforce/train_lab.py \\
    --task Isaac-Humanoid-v0 \\
    --headless

# Evaluate trained policy
python scripts/renforce/play_lab.py \\
    --task Isaac-Humanoid-v0 \\
    --checkpoint runs/Isaac-Humanoid-v0/model.pth
```

---

## 2. VLA Training Scripts: `vla/` ⭐

**New Structure** for VLA (System 1 + System 2) training

```
vla/
├── __init__.py
├── pretrain/                     # VLA pretraining
│   ├── __init__.py
│   ├── train_single_gpu.py      # Single-GPU pretrain ⭐
│   ├── train_ddp.py             # Multi-GPU DDP pretrain ⭐
│   └── train_deepspeed.py       # DeepSpeed pretrain (Future)
├── post_train/                   # VLA post-training
│   ├── __init__.py
│   ├── train_sft.py             # SFT training ⭐
│   └── train_dpo.py             # DPO training (Future)
├── rl/                           # VLA RL fine-tuning
│   ├── __init__.py
│   ├── train_ppo_finetune.py    # PPO fine-tuning ⭐
│   └── train_sac_finetune.py    # SAC fine-tuning
├── eval/                         # VLA evaluation
│   ├── __init__.py
│   ├── evaluate_pretrain.py     # Evaluate pretrained VLA
│   └── evaluate_rl.py           # Evaluate RL-finetuned VLA
└── utils/                        # VLA utilities
    ├── __init__.py
    ├── visualize_predictions.py # Visualize VLA predictions
    └── export_model.py          # Export VLA (ONNX/TorchScript)
```

**Total: 21 files** (15 .py + 6 __init__.py)

---

### 2.1 Pretrain Scripts: `vla/pretrain/`

#### File: `train_single_gpu.py` (Phase 3, Week 3, Priority P0)

**Purpose**: Single-GPU VLA pretraining for quick validation

```python
#!/usr/bin/env python3
"""
Single-GPU VLA Pretraining Script

Usage:
    python scripts/vla/pretrain/train_single_gpu.py \\
        --config RRF_vla_tasks.vla_pretrain.minimal_example \\
        --log_dir logs/pretrain_single

Features:
- Single GPU training
- Mixed precision (FP16/BF16)
- Checkpoint saving
- TensorBoard logging
"""

import argparse
import torch
from pathlib import Path

from RoboRenForce.utils.package import load_config
from RoboRenForce.runners.vla.pretrain import VLAPretrainRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU VLA Pretraining")
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config module path (e.g., RRF_vla_tasks.vla_pretrain.minimal_example)",
    )
    
    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override dataset root path",
    )
    
    # Training
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    
    # Logging
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/pretrain_single",
        help="Logging directory",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    
    # Checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    
    # Override config with CLI args
    if args.data_root is not None:
        cfg.dataset_cfg.data_root = args.data_root
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    
    # Validate config
    missing_fields = cfg.validate()
    if missing_fields:
        raise ValueError(f"Config missing required fields: {missing_fields}")
    
    # Create runner
    print(f"Creating VLA Pretrain Runner...")
    runner = VLAPretrainRunner(
        cfg=cfg,
        log_dir=args.log_dir,
        device=args.device,
    )
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        runner.load_checkpoint(args.resume)
    
    # Train
    print(f"Starting training for {cfg.num_epochs} epochs...")
    runner.learn(num_epochs=cfg.num_epochs)
    
    print("Training complete!")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")


if __name__ == "__main__":
    main()
```

---

#### File: `train_ddp.py` (Phase 4, Week 4, Priority P0) ⭐

**Purpose**: Multi-GPU DDP VLA pretraining (8-16 GPUs)

```python
#!/usr/bin/env python3
"""
Multi-GPU DDP VLA Pretraining Script

Usage:
    # 8 GPUs on single node
    torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \\
        --config RRF_vla_tasks.vla_pretrain.humanoid_qwen3vl \\
        --log_dir logs/pretrain_ddp

    # 16 GPUs on 2 nodes
    torchrun --nnodes=2 --nproc_per_node=8 \\
        --master_addr=<master_ip> --master_port=29500 \\
        scripts/vla/pretrain/train_ddp.py \\
        --config RRF_vla_tasks.vla_pretrain.humanoid_qwen3vl

Features:
- Multi-GPU training with PyTorch DDP
- Gradient synchronization
- Distributed data sampling
- Checkpoint saving on rank 0
"""

import argparse
import os
import torch
import torch.distributed as dist
from pathlib import Path

from RoboRenForce.utils.package import load_config
from RoboRenForce.runners.vla.pretrain import DistributedVLAPretrainRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU DDP VLA Pretraining")
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config module path",
    )
    
    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override dataset root path",
    )
    
    # Training
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size (per GPU)",
    )
    
    # Logging
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/pretrain_ddp",
        help="Logging directory",
    )
    
    # Checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    
    # DDP settings (usually set by torchrun)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank (set by torchrun)",
    )
    
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training."""
    # torchrun sets these environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size, rank


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def main():
    args = parse_args()
    
    # Setup distributed
    local_rank, world_size, rank = setup_distributed()
    
    if rank == 0:
        print(f"Distributed training with {world_size} GPUs")
        print(f"Loading config: {args.config}")
    
    # Load config
    cfg = load_config(args.config)
    
    # Override config with CLI args
    if args.data_root is not None:
        cfg.dataset_cfg.data_root = args.data_root
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size  # Per GPU
    
    # Validate config (only on rank 0)
    if rank == 0:
        missing_fields = cfg.validate()
        if missing_fields:
            raise ValueError(f"Config missing required fields: {missing_fields}")
    
    # Synchronize before creating runner
    dist.barrier()
    
    # Create distributed runner
    if rank == 0:
        print(f"Creating Distributed VLA Pretrain Runner...")
    
    runner = DistributedVLAPretrainRunner(
        cfg=cfg,
        log_dir=args.log_dir,
        device=f"cuda:{local_rank}",
    )
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
        runner.load_checkpoint(args.resume)
    
    # Train
    if rank == 0:
        print(f"Starting DDP training for {cfg.num_epochs} epochs...")
        print(f"Effective batch size: {cfg.batch_size * world_size}")
    
    runner.learn(num_epochs=cfg.num_epochs)
    
    # Cleanup
    if rank == 0:
        print("Training complete!")
        print(f"Checkpoints saved to: {cfg.checkpoint_dir}")
        print(f"Logs saved to: {args.log_dir}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
```

---

#### File: `train_deepspeed.py` (Phase 7+, Priority P2)

**Purpose**: Large-scale training with DeepSpeed (64+ GPUs, model parallelism)

```python
#!/usr/bin/env python3
"""
DeepSpeed VLA Pretraining Script

Usage:
    deepspeed --num_gpus=64 scripts/vla/pretrain/train_deepspeed.py \\
        --config RRF_vla_tasks.vla_pretrain.large_scale \\
        --deepspeed_config deepspeed_config.json

Features:
- ZeRO optimization (stage 2/3)
- Gradient accumulation
- Pipeline parallelism (optional)
- Large model support (>10B parameters)
"""

# Future implementation
# Reference: .references/lerobot/lerobot/scripts/train_deepspeed.py
```

---

### 2.2 Post-Train Scripts: `vla/post_train/`

#### File: `train_sft.py` (Phase 5, Week 5, Priority P1) ⭐

**Purpose**: Supervised Fine-Tuning on task-specific data

```python
#!/usr/bin/env python3
"""
SFT Training Script

Usage:
    torchrun --nproc_per_node=4 scripts/vla/post_train/train_sft.py \\
        --config RRF_vla_tasks.vla_post_train.sft_humanoid_reach \\
        --pretrained checkpoints/humanoid_qwen3vl/vla_step_10000.pth \\
        --log_dir logs/sft_reach

Features:
- Load pretrained VLA
- Fine-tune on task-specific demos
- Lower learning rate than pretraining
- Supports DDP (4 GPUs typical)
"""

import argparse
import os
import torch
import torch.distributed as dist

from RoboRenForce.utils.package import load_config
from RoboRenForce.runners.vla.post_train import VLASFTRunner, DistributedVLASFTRunner


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training")
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config module path",
    )
    
    # Pretrained model
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to pretrained VLA checkpoint",
    )
    
    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override dataset root path",
    )
    
    # Training
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    
    # Logging
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/sft",
        help="Logging directory",
    )
    
    # Distributed
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use DDP (auto-detected if torchrun)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check if running with DDP
    if "LOCAL_RANK" in os.environ:
        args.distributed = True
    
    # Setup distributed if needed
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = int(os.environ["RANK"])
    else:
        device = "cuda"
        rank = 0
    
    # Load config
    if rank == 0:
        print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    
    # Set pretrained path
    cfg.pretrained_vla_path = args.pretrained
    
    # Override config
    if args.data_root is not None:
        cfg.dataset_cfg.data_root = args.data_root
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    
    # Create runner
    if args.distributed:
        runner = DistributedVLASFTRunner(cfg, args.log_dir, device)
    else:
        runner = VLASFTRunner(cfg, args.log_dir, device)
    
    # Train
    if rank == 0:
        print(f"Starting SFT training...")
    runner.learn(num_epochs=cfg.num_epochs)
    
    # Cleanup
    if args.distributed:
        dist.destroy_process_group()
    
    if rank == 0:
        print("SFT training complete!")


if __name__ == "__main__":
    main()
```

---

#### File: `train_dpo.py` (Phase 7+, Priority P2)

**Purpose**: Direct Preference Optimization (future)

```python
#!/usr/bin/env python3
"""
DPO Training Script

Usage:
    torchrun --nproc_per_node=4 scripts/vla/post_train/train_dpo.py \\
        --config RRF_vla_tasks.vla_post_train.dpo_humanoid \\
        --pretrained checkpoints/sft_reach/vla_step_5000.pth

Features:
- Preference-based fine-tuning
- Requires paired preference dataset
"""

# Future implementation
# Reference: https://arxiv.org/abs/2305.18290
```

---

### 2.3 RL Fine-tuning Scripts: `vla/rl/`

#### File: `train_ppo_finetune.py` (Phase 6, Week 6, Priority P1) ⭐

**Purpose**: RL fine-tuning with PPO

```python
#!/usr/bin/env python3
"""
PPO RL Fine-tuning Script

Usage:
    python scripts/vla/rl/train_ppo_finetune.py \\
        --config RRF_vla_tasks.vla_rl_finetune.ppo_humanoid_reach \\
        --pretrained checkpoints/sft_reach/vla_step_5000.pth \\
        --task Isaac-Humanoid-Reach-v0 \\
        --num_envs 4096 \\
        --log_dir logs/rl_finetune

Features:
- Load pretrained/SFT VLA
- Apply LoRA for efficient fine-tuning
- Run PPO with environment parallelism (4096 envs)
- Freeze VLM backbone, train action head + fusion
"""

import argparse
import torch

from RoboRenForce.utils.package import load_config
from RoboRenForce.runners.vla.rl import VLARLRunner


def parse_args():
    parser = argparse.ArgumentParser(description="PPO RL Fine-tuning")
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config module path",
    )
    
    # Pretrained model
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to pretrained/SFT VLA checkpoint",
    )
    
    # Environment
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Isaac Lab task name",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4096,
        help="Number of parallel environments",
    )
    
    # Training
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Override max iterations",
    )
    
    # Logging
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/rl_finetune",
        help="Logging directory",
    )
    
    # LoRA
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA (fine-tune all parameters)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    
    # Set pretrained path
    cfg.pretrained_vla_path = args.pretrained
    
    # Override config
    cfg.task = args.task
    cfg.num_envs = args.num_envs
    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations
    if args.no_lora:
        cfg.rl_algorithm_cfg.use_lora = False
    
    # Validate config
    missing_fields = cfg.validate()
    if missing_fields:
        raise ValueError(f"Config missing required fields: {missing_fields}")
    
    # Create runner
    print(f"Creating VLA RL Runner...")
    print(f"Task: {cfg.task}")
    print(f"Num envs: {cfg.num_envs}")
    print(f"LoRA enabled: {cfg.rl_algorithm_cfg.use_lora}")
    
    runner = VLARLRunner(
        cfg=cfg,
        log_dir=args.log_dir,
        device="cuda",
    )
    
    # Train
    print(f"Starting RL fine-tuning for {cfg.max_iterations} iterations...")
    runner.learn(num_iterations=cfg.max_iterations)
    
    print("RL fine-tuning complete!")
    print(f"Final checkpoint: {cfg.checkpoint_dir}/vla_rl_final.pth")


if __name__ == "__main__":
    main()
```

---

#### File: `train_sac_finetune.py` (Phase 7+, Priority P2)

**Purpose**: RL fine-tuning with SAC (off-policy)

```python
#!/usr/bin/env python3
"""
SAC RL Fine-tuning Script

Similar to train_ppo_finetune.py but uses SAC algorithm.
Useful for continuous control tasks with replay buffer.
"""

# Future implementation
```

---

### 2.4 Evaluation Scripts: `vla/eval/`

#### File: `evaluate_pretrain.py` (Phase 7+, Priority P2)

**Purpose**: Evaluate pretrained VLA on validation set

```python
#!/usr/bin/env python3
"""
Evaluate Pretrained VLA

Usage:
    python scripts/vla/eval/evaluate_pretrain.py \\
        --vla_path checkpoints/humanoid_qwen3vl/vla_step_10000.pth \\
        --data_root data/humanoid_mixed_tasks \\
        --split val \\
        --output_dir results/pretrain_eval

Metrics:
- Action MSE
- Success rate (if available)
- Visualization of predictions
"""

# Implementation for offline evaluation
```

---

#### File: `evaluate_rl.py` (Phase 7+, Priority P2)

**Purpose**: Evaluate RL-finetuned VLA in simulation

```python
#!/usr/bin/env python3
"""
Evaluate RL-Finetuned VLA

Usage:
    python scripts/vla/eval/evaluate_rl.py \\
        --vla_path checkpoints/rl_finetune_reach/vla_rl_final.pth \\
        --task Isaac-Humanoid-Reach-v0 \\
        --num_episodes 100 \\
        --output_dir results/rl_eval

Metrics:
- Average reward
- Success rate
- Episode length
- Videos of rollouts
"""

# Implementation for online evaluation in Isaac Lab
```

---

### 2.5 Utility Scripts: `vla/utils/`

#### File: `visualize_predictions.py` (Phase 7+, Priority P2)

**Purpose**: Visualize VLA action predictions

```python
#!/usr/bin/env python3
"""
Visualize VLA Predictions

Usage:
    python scripts/vla/utils/visualize_predictions.py \\
        --vla_path checkpoints/vla.pth \\
        --data_root data/humanoid_demos \\
        --output_dir visualizations \\
        --num_samples 50

Outputs:
- Overlay predicted actions on images
- Plot action trajectories
- Compare with ground truth
"""

# Implementation for visualization
```

---

#### File: `export_model.py` (Phase 7+, Priority P2)

**Purpose**: Export VLA to ONNX/TorchScript for deployment

```python
#!/usr/bin/env python3
"""
Export VLA Model

Usage:
    # Export to ONNX
    python scripts/vla/utils/export_model.py \\
        --vla_path checkpoints/vla.pth \\
        --format onnx \\
        --output exported_vla.onnx

    # Export to TorchScript
    python scripts/vla/utils/export_model.py \\
        --vla_path checkpoints/vla.pth \\
        --format torchscript \\
        --output exported_vla.pt

Supported formats:
- ONNX (for cross-platform deployment)
- TorchScript (for C++ deployment)
"""

# Implementation for model export
```

---

## 3. Distributed Launchers: `distributed/` ⭐

**New Scripts** for distributed training utilities

```
distributed/
├── __init__.py
├── launch_ddp.py              # DDP launcher helper
├── launch_slurm.py            # SLURM job generator
└── launch_torchrun.py         # torchrun wrapper
```

**Total: 4 files**

---

### File: `launch_ddp.py` (Phase 4, Week 4, Priority P1)

**Purpose**: Helper script for launching DDP training

```python
#!/usr/bin/env python3
"""
DDP Launcher Helper

Usage:
    python scripts/distributed/launch_ddp.py \\
        --num_gpus 8 \\
        --script scripts/vla/pretrain/train_ddp.py \\
        -- --config RRF_vla_tasks.vla_pretrain.humanoid_qwen3vl

Features:
- Automatically sets up torchrun command
- Handles multi-node setup
- Port allocation
"""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="DDP Launcher")
    
    # DDP settings
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs per node",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Master node address (for multi-node)",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Node rank (0 for master)",
    )
    
    # Script to launch
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Training script to launch",
    )
    
    # Pass-through args for the training script
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments for training script (after --)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        f"--nnodes={args.num_nodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        args.script,
    ]
    
    # Add script arguments (remove leading -- if present)
    if args.script_args:
        if args.script_args[0] == "--":
            script_args = args.script_args[1:]
        else:
            script_args = args.script_args
        cmd.extend(script_args)
    
    # Print command
    print("Launching DDP training:")
    print(" ".join(cmd))
    print()
    
    # Run
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
```

---

### File: `launch_slurm.py` (Phase 4, Week 4, Priority P1)

**Purpose**: Generate SLURM job scripts for cluster training

```python
#!/usr/bin/env python3
"""
SLURM Job Generator

Usage:
    python scripts/distributed/launch_slurm.py \\
        --nodes 8 \\
        --gpus_per_node 8 \\
        --script scripts/vla/pretrain/train_ddp.py \\
        --config RRF_vla_tasks.vla_pretrain.humanoid_qwen3vl \\
        --output slurm_job.sh

Then submit:
    sbatch slurm_job.sh
"""

import argparse


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --output={output_log}
#SBATCH --error={error_log}

# Setup environment
module load cuda/12.1
source ~/.bashrc
conda activate roborenforce

# Master node address
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch training
srun torchrun \\
    --nnodes=$SLURM_NNODES \\
    --nproc_per_node={gpus_per_node} \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    {script} {script_args}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="SLURM Job Generator")
    
    # SLURM settings
    parser.add_argument("--job_name", type=str, default="vla_pretrain")
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument("--time", type=str, default="48:00:00")
    parser.add_argument("--partition", type=str, default="gpu")
    parser.add_argument("--output_log", type=str, default="logs/slurm_%j.out")
    parser.add_argument("--error_log", type=str, default="logs/slurm_%j.err")
    
    # Training script
    parser.add_argument("--script", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    
    # Output
    parser.add_argument("--output", type=str, default="slurm_job.sh")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build script args
    script_args = f"--config {args.config}"
    
    # Fill template
    job_script = SLURM_TEMPLATE.format(
        job_name=args.job_name,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        time=args.time,
        partition=args.partition,
        output_log=args.output_log,
        error_log=args.error_log,
        script=args.script,
        script_args=script_args,
    )
    
    # Write to file
    with open(args.output, "w") as f:
        f.write(job_script)
    
    print(f"SLURM job script written to: {args.output}")
    print(f"Submit with: sbatch {args.output}")


if __name__ == "__main__":
    main()
```

---

### File: `launch_torchrun.py` (Phase 4, Week 4, Priority P2)

**Purpose**: Wrapper for torchrun with better defaults

```python
#!/usr/bin/env python3
"""
Torchrun Wrapper

Simplified torchrun interface with sensible defaults.

Usage:
    python scripts/distributed/launch_torchrun.py \\
        scripts/vla/pretrain/train_ddp.py \\
        --config RRF_vla_tasks.vla_pretrain.humanoid_qwen3vl
"""

# Thin wrapper around torchrun
```

---

## 4. Data Processing Scripts: `data/`

**Existing scripts** (keep as is) + **New VLA data scripts**

```
data/
├── __init__.py
├── rlds_to_lerobot.py            # Convert RLDS to LeRobot format (NEW) ⭐
├── isaaclab_to_lerobot.py        # Convert Isaac Lab demos to LeRobot (NEW) ⭐
├── compute_dataset_stats.py      # Compute normalization stats (NEW)
├── merge_datasets.py             # Merge multiple datasets (NEW)
└── visualize_dataset.py          # Visualize LeRobot dataset (NEW)
```

**Total: 6 files** (5 new + 1 __init__.py)

These scripts are for **Phase 0-1** (Data Preparation, Priority P0)

### File: `rlds_to_lerobot.py` (Phase 0, Priority P0)

**Purpose**: Convert RLDS format (RoboTwin, Open X-Embodiment) to LeRobot

```python
#!/usr/bin/env python3
"""
RLDS to LeRobot Converter

Usage:
    python scripts/data/rlds_to_lerobot.py \\
        --input_dir data/rlds/fractal20220817_data \\
        --output_dir data/lerobot/fractal \\
        --robot_type fractal \\
        --video_codec libsvtav1

Reference: .references/lerobot/lerobot/scripts/push_dataset_to_hub.py
"""

# Implementation for RLDS → LeRobot conversion
```

---

### File: `isaaclab_to_lerobot.py` (Phase 0, Priority P0)

**Purpose**: Convert Isaac Lab demonstrations to LeRobot format

```python
#!/usr/bin/env python3
"""
Isaac Lab to LeRobot Converter

Usage:
    python scripts/data/isaaclab_to_lerobot.py \\
        --input_dir data/isaaclab_demos/humanoid_reach \\
        --output_dir data/lerobot/humanoid_reach \\
        --robot_type humanoid \\
        --fps 30

Converts:
- Isaac Lab HDF5/pickle demos → LeRobot Parquet + videos
- Extracts: image, proprioception, action, reward, done
"""

# Implementation for Isaac Lab → LeRobot conversion
```

---

## 5. Third-Party Scripts: `third_party/`

**Existing scripts** (unchanged)

```
third_party/
├── isaaclab/
│   ├── test_env.py
│   └── visualize_fk_from_pkl.py
└── rename_repo.sh
```

---

## Priority Summary

### Phase 0 (Data Prep): **2 files** (P0)
- ✅ `data/rlds_to_lerobot.py`
- ✅ `data/isaaclab_to_lerobot.py`

### Phase 3 (Week 3): **2 files** (P0)
- ✅ `vla/pretrain/train_single_gpu.py`
- ✅ `vla/pretrain/__init__.py`

### Phase 4 (Week 4): **5 files** (P0/P1)
- ✅ `vla/pretrain/train_ddp.py` (P0)
- ✅ `distributed/launch_ddp.py` (P1)
- ✅ `distributed/launch_slurm.py` (P1)
- ✅ `distributed/__init__.py` (P1)

### Phase 5 (Week 5): **2 files** (P1)
- ✅ `vla/post_train/train_sft.py`
- ✅ `vla/post_train/__init__.py`

### Phase 6 (Week 6): **2 files** (P1)
- ✅ `vla/rl/train_ppo_finetune.py`
- ✅ `vla/rl/__init__.py`

### Phase 7+ (Future): **8 files** (P2)
- `vla/pretrain/train_deepspeed.py`
- `vla/post_train/train_dpo.py`
- `vla/rl/train_sac_finetune.py`
- `vla/eval/evaluate_pretrain.py`
- `vla/eval/evaluate_rl.py`
- `vla/utils/visualize_predictions.py`
- `vla/utils/export_model.py`
- `distributed/launch_torchrun.py`

---

## Total Script Count

- **VLA Scripts**: 21 files (15 .py + 6 __init__.py)
- **Distributed Scripts**: 4 files
- **Data Scripts**: 6 files
- **RL Scripts (existing)**: 3 files (unchanged)
- **Third-party (existing)**: 3 files (unchanged)

**Total New Scripts**: 31 files (21 VLA + 4 distributed + 6 data)

**Core Scripts (P0)**: 5 files
- train_single_gpu.py
- train_ddp.py
- rlds_to_lerobot.py
- isaaclab_to_lerobot.py
