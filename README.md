# RoboRenForce

<div align="center">

**Modular RL & VLA Framework for Robotics — from Locomotion to Vision-Language-Action**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3-green.svg)](LICENSE)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-0.21+-orange.svg)](https://github.com/isaac-sim/Isaac-Lab)
[![MJLab](https://img.shields.io/badge/MJLab-MuJoCo%20Warp-purple.svg)](https://github.com/mujocolab/mjlab)

</div>

RoboRenForce is a unified framework that covers the full robotics RL pipeline: classic locomotion control (PPO/SAC on Isaac Lab, MJLab, Gymnasium), vision-language-action model training (Qwen2-VL, Qwen3-VL, OpenPI, GR00T), and multi-stage learning (pretrain → SFT → RL fine-tuning). Everything is driven by a composable `@configclass` system and a consistent wrapper chain across simulators.

---

## Architecture

```
┌─────────────────────────── RoboRenForce ───────────────────────────┐
│                                                                     │
│  System 2 (VLM Backbone)     System 1 (Action Expert)     System 0 │
│  ┌───────────────────┐       ┌──────────────────┐       ┌────────┐ │
│  │ Qwen2-VL / Qwen3  │──────▶│ Regression Head  │──────▶│ Loco   │ │
│  │ OpenPI / GR00T    │       │ Diffusion Head   │       │ Policy │ │
│  │ (frozen / LoRA)   │       │ Flow-Match Head  │       │ (Psi0) │ │
│  └───────────────────┘       └──────────────────┘       └────────┘ │
│           ▲                          ▲                       ▲      │
│     observations               configclass               env API   │
│           │                          │                       │      │
│  ┌────────┴──────────────────────────┴───────────────────────┴────┐ │
│  │              Environment Wrapper Chain                         │ │
│  │  Isaac Lab ─┐                                                 │ │
│  │  MJLab     ─┤─▶ VecEnv ─▶ DynamicEnv ─▶ GroupVecWrapper     │ │
│  │  RoboTwin  ─┤                                                 │ │
│  │  Gymnasium ─┘                                                 │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone <repository-url>
cd RoboRenForce

# Core framework
pip install -e source/RoboRenForce

# Task packages (install what you need)
pip install -e source/tasks/RRF_isaaclab    # Isaac Lab locomotion/manipulation
pip install -e source/tasks/RRF_mjlab       # MJLab (MuJoCo Warp) locomotion
pip install -e source/tasks/RRF_robotwin    # RoboTwin manipulation
pip install -e source/tasks/RRF_humanoid_psi0  # Humanoid offline tasks

# External setup (robot assets, etc.)
bash scripts/setup_ext.sh
```

<details>
<summary><b>VLA model setup (optional)</b></summary>

```bash
# Download VLM weights
bash scripts/models/setup_models.sh qwen2vl    # Qwen2-VL 2B (~4.2GB)
bash scripts/models/setup_models.sh qwen3vl    # Qwen3-VL 2B
bash scripts/models/setup_models.sh openpi     # OpenPI pi0.5 4B
bash scripts/models/setup_models.sh groot      # GR00T N1.7 3B

# Or install dependencies only
pip install "transformers>=4.37" qwen-vl-utils accelerate peft
```

See [docs/models.md](docs/models.md) for per-model details, VRAM requirements, and usage examples.

</details>

<details>
<summary><b>MJLab simulator setup</b></summary>

```bash
# Clone and install MJLab
git clone https://github.com/mujocolab/mjlab.git
pip install -e mjlab

# Requires: mujoco>=3.7.0, mujoco-warp>=3.7.0.1, warp-lang>=1.12.0
```

</details>

---

## Quick Start

### Locomotion Training (MJLab)

```bash
# Go1 quadruped on flat terrain — PPO
python scripts/renforce/train_mjlab.py \
    --task Mjlab-Velocity-Flat-Unitree-Go1 \
    --num_envs 4096 --device cuda:0

# G1 humanoid on rough terrain
python scripts/renforce/train_mjlab.py \
    --task Mjlab-Velocity-Rough-Unitree-G1 \
    --num_envs 2048 --max_iterations 30000
```

### Locomotion Training (Isaac Lab)

```bash
python scripts/renforce/train_lab.py \
    --task RoboRenForce-AFR-UnitreeGo1Flat-PPO \
    --num_envs 4096 --headless
```

### VLA Pretraining

```bash
# Single GPU
python scripts/vla/pretrain/train_single_gpu.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --dataset_path data/my_dataset --epochs 20

# Multi-GPU DDP
torchrun --nproc_per_node=4 scripts/vla/pretrain/train_ddp.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --dataset_path data/my_dataset --epochs 20
```

### VLA RL Fine-tuning (GRPO / PPO)

```bash
python scripts/vla/rl/train_robotwin_grpo.py \
    --task close_laptop_lid --algo grpo --num_envs 32
```

<details>
<summary><b>More examples: SFT, evaluation, data pipeline</b></summary>

**Supervised Fine-tuning (SFT)**
```bash
# Single GPU
python scripts/vla/post_train/train_sft.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --dataset_path data/my_dataset --epochs 5

# Multi-GPU
torchrun --nproc_per_node=2 scripts/vla/post_train/train_sft_ddp.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --dataset_path data/my_dataset
```

**Evaluation / Playback**
```bash
# MJLab
python scripts/renforce/play_mjlab.py \
    --target logs/RFRL/mjlab_go1/model_5000.pt --num_envs 64

# Isaac Lab
python scripts/renforce/play_lab.py \
    --target logs/RFRL/go1_ppo/model_5000.pt --video
```

**Data Conversion**
```bash
# Download LeRobot dataset
python scripts/data/download_lerobot_dataset.py --repo lerobot/aloha_sim

# Isaac Lab trajectories → LeRobot format
python scripts/data/isaaclab_to_lerobot.py --input traj/ --output data/lerobot/

# RLDS → LeRobot
python scripts/data/rlds_to_lerobot.py --input rlds_data/ --output data/lerobot/
```

</details>

---

## Algorithms

| Category | Algorithm | Runner | Reference |
|----------|-----------|--------|-----------|
| **On-Policy** | PPO | `OnPolicyRunner` | [ppo.py](source/RoboRenForce/RoboRenForce/algorithms/on_policy/ppo.py) |
| | CAPS-PPO / L2C2-PPO / Lips-PPO | `OnPolicyRunner` | [smooth.py](source/RoboRenForce/RoboRenForce/algorithms/on_policy/smooth.py) |
| | SAPG-PPO | `SAPGOnPolicyRunner` | [sapg/](source/RoboRenForce/RoboRenForce/algorithms/on_policy/sapg/) |
| | MBPO | `MBPOOnPolicyRunner` | [mbpo/](source/RoboRenForce/RoboRenForce/algorithms/on_policy/mbpo/) |
| **Off-Policy** | SAC | `OffPolicyRunner` | [sac/](source/RoboRenForce/RoboRenForce/algorithms/off_policy/sac/) |
| | DSAC | `OffPolicyRunner` | [dsac/](source/RoboRenForce/RoboRenForce/algorithms/off_policy/dsac/) |
| **VLA Training** | Pretrain (SL) | `VLAPretrainRunner` | [pretrain_algorithm.py](source/RoboRenForce/RoboRenForce/algorithms/vla_training/pretrain_algorithm.py) |
| | SFT (KL reg.) | `VLASFTRunner` | [sft.py](source/RoboRenForce/RoboRenForce/algorithms/vla_training/sft.py) |
| | GRPO | `VLAGRPORunner` | [grpo.py](source/RoboRenForce/RoboRenForce/algorithms/vla_training/grpo.py) |
| | PPO (GAE) | `VLAPPORunner` | [ppo.py](source/RoboRenForce/RoboRenForce/algorithms/vla_training/ppo.py) |
| | IQL / DAgger / SAC | — | [iql.py](source/RoboRenForce/RoboRenForce/algorithms/vla_training/iql.py), [dagger.py](source/RoboRenForce/RoboRenForce/algorithms/vla_training/dagger.py), [sac.py](source/RoboRenForce/RoboRenForce/algorithms/vla_training/sac.py) |
| **World Model** | Dynamics / Flow | `WorldModelBasedRunner` | [world_model/](source/RoboRenForce/RoboRenForce/runners/world_model/) |
| **Imitation** | Distillation | `DistillationRunner` | [imitation/](source/RoboRenForce/RoboRenForce/runners/imitation/) |

---

## Supported VLM Backbones

| Model | Params | Output Dim | Action Heads | License |
|-------|--------|-----------|--------------|---------|
| [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | 2B / 7B | 1536 | Regression, Diffusion | Apache 2.0 |
| [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 2B / 8B | 2048 | Regression, Diffusion | Apache 2.0 |
| [OpenPI (pi0.5)](https://huggingface.co/lerobot/pi05_base) | 4B | 2048 | Flow Matching | Apache 2.0 + Gemma |
| [GR00T N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B) | 3B | 2048 | DiT | Apache 2.0 |
| MLP Baseline | ~1M | 64 | Regression | Built-in |

See [docs/models.md](docs/models.md) for full details.

---

## Supported Environments

| Platform | Package | Robots / Tasks | Type | Script |
|----------|---------|----------------|------|--------|
| **MJLab** (MuJoCo Warp) | [`RRF_mjlab`](source/tasks/RRF_mjlab/) | Go1, G1 — velocity tracking (flat/rough) | Locomotion | [`train_mjlab.py`](scripts/renforce/train_mjlab.py) |
| **Isaac Lab** (Isaac Sim) | [`RRF_isaaclab`](source/tasks/RRF_isaaclab/) | A1, Go1, Go2, Anymal B/C/D, H1, G1 | Locomotion | [`train_lab.py`](scripts/renforce/train_lab.py) |
| **RoboTwin** (SAPIEN3) | [`RRF_robotwin`](source/tasks/RRF_robotwin/) | Piper, ALOHA — 60+ manipulation tasks | VLA RL | [`train_robotwin_grpo.py`](scripts/vla/rl/train_robotwin_grpo.py) |
| **LIBERO** | [`RRF_libero`](source/tasks/RRF_libero/) | Franka — 10/90/130 manipulation tasks | VLA Pretrain/RL | [`train_vla_benchmark.py`](scripts/vla/rl/train_vla_benchmark.py) |
| **ManiSkill** | [`RRF_maniskill`](source/tasks/RRF_maniskill/) | Franka — GPU-accelerated manipulation | VLA Pretrain/RL | [`train_vla_benchmark.py`](scripts/vla/rl/train_vla_benchmark.py) |
| **CALVIN** | [`RRF_calvin`](source/tasks/RRF_calvin/) | Franka — 5-subtask long-horizon eval | VLA Pretrain/RL | [`train_vla_benchmark.py`](scripts/vla/rl/train_vla_benchmark.py) |
| **D4RL** | [`RRF_d4rl`](source/tasks/RRF_d4rl/) | Walker2d, Hopper, HalfCheetah | Offline RL | [`train_vla_benchmark.py`](scripts/vla/rl/train_vla_benchmark.py) |
| **Humanoid Psi0** | [`RRF_humanoid_psi0`](source/tasks/RRF_humanoid_psi0/) | G1 Dex3 — 14 offline tasks | Offline | Offline runner |
| **Gymnasium** | Built-in | Classic control, MuJoCo | RL Baseline | [`train_gym.py`](scripts/renforce/train_gym.py) |

---

## Configuration System

All components are configured via `@configclass` — a decorator that extends Python dataclasses with type validation, serialization, and factory construction.

<details>
<summary><b>Example: defining a custom PPO config</b></summary>

```python
from RoboRenForce import configclass
from RoboRenForce import runners, algorithms, components, networks

@configclass
class MyLocoPPOCfg(runners.OnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    experiment_name = "my_experiment"

    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            ),
            use_log_std=False
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            )
        )
    )

    algorithm = algorithms.PPOCfg(
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
    )
```

</details>

<details>
<summary><b>Example: registering a task with gymnasium</b></summary>

```python
import gymnasium as gym

gym.register(
    id="RoboRenForce-MyTask-PPO",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": my_env_cfg,
        "RoboRenForce_entry_point": MyLocoPPOCfg(),
    },
)
```

</details>

<details>
<summary><b>Example: environment wrapper chain</b></summary>

```python
from mjlab.envs import ManagerBasedRlEnv
from RRF_mjlab_tasks.mjlab_utils import (
    RoboRenForceMJLabEnvWrapper,   # Base: obs remapping, step adaptation
    MJLabDynamicEnvWrapper,         # + reward/command extraction, dim_params
    MJLabGroupVecWrapper,           # + train/eval env partitioning
)

env = ManagerBasedRlEnv(cfg=my_cfg, device="cuda:0")
wrapped = MJLabDynamicEnvWrapper(env)

print(wrapped.num_envs)       # 4096
print(wrapped.dim_params)     # {'policy_dim': 48, 'critic_dim': 72, ...}
obs, extras = wrapped.reset() # obs: (4096, 48)
```

</details>

---

## Project Structure

```
RoboRenForce/
├── source/
│   ├── RoboRenForce/                   # Core framework
│   │   └── RoboRenForce/
│   │       ├── algorithms/             # RL algorithms
│   │       │   ├── on_policy/          #   PPO, MBPO, SAPG, smooth variants
│   │       │   ├── off_policy/         #   SAC, DSAC
│   │       │   ├── vla_training/       #   Pretrain, SFT, GRPO, PPO, IQL, DAgger
│   │       │   └── world_model_trainer/
│   │       ├── runners/                # Training loops
│   │       │   ├── on_policy/          #   OnPolicyRunner, SAPG, EPO
│   │       │   ├── off_policy/         #   OffPolicyRunner
│   │       │   ├── vla/               #   Pretrain, SFT, GRPO (+ DDP variants)
│   │       │   └── world_model/        #   MBPO, Flow model
│   │       ├── networks/               # Neural network modules
│   │       │   ├── vlm/               #   Qwen2-VL, Qwen3-VL, OpenPI, GR00T
│   │       │   ├── transformer/       #   Transformer backbone
│   │       │   └── mlp.py, vae/, moe.py, fft_filter.py
│   │       ├── components/             # Actors, critics, normalizers
│   │       │   ├── actor/             #   Gaussian, SAC, Lipschitz, VLA actors
│   │       │   ├── critic/            #   V-net, Q-net, distributional
│   │       │   └── normalizer/        #   Empirical normalizer
│   │       ├── buffer/                 # Replay buffers & rollout storage
│   │       └── utils/                  # Config system, env wrappers, tools
│   │           ├── configclass/       #   @configclass decorator
│   │           └── env_wrapper/       #   Lab, Gym, VLA wrapper chains
│   └── tasks/                          # Task packages
│       ├── RRF_isaaclab/              #   Isaac Lab locomotion & manipulation
│       ├── RRF_mjlab/                 #   MJLab (MuJoCo Warp) locomotion
│       ├── RRF_robotwin/              #   RoboTwin 60+ manipulation tasks
│       └── RRF_humanoid_psi0/         #   Humanoid offline datasets
├── scripts/
│   ├── renforce/                       # Standard RL training & evaluation
│   │   ├── train_lab.py               #   Isaac Lab training
│   │   ├── train_mjlab.py             #   MJLab training
│   │   ├── train_gym.py               #   Gymnasium training
│   │   ├── play_lab.py                #   Isaac Lab evaluation
│   │   └── play_mjlab.py              #   MJLab evaluation
│   ├── vla/                            # VLA model training
│   │   ├── pretrain/                  #   Single-GPU & DDP pretraining
│   │   ├── post_train/                #   SFT (single & DDP)
│   │   └── rl/                        #   GRPO / PPO fine-tuning
│   └── data/                           # Dataset tools
│       ├── download_lerobot_dataset.py
│       ├── isaaclab_to_lerobot.py
│       └── rlds_to_lerobot.py
├── docs/
│   ├── models.md                       # VLM backbone details & setup
│   └── BENCHMARK_PLAN.md              # Benchmark experiment specs
└── tests/                              # Test suites
```

---

## Multi-GPU Training (DDP)

RoboRenForce supports distributed training via PyTorch DDP for VLA workloads:

```bash
# VLA Pretraining — 4 GPUs
torchrun --nproc_per_node=4 scripts/vla/pretrain/train_ddp.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --dataset_path data/humanoid_psi0 --epochs 20

# VLA SFT — 2 GPUs
torchrun --nproc_per_node=2 scripts/vla/post_train/train_sft_ddp.py \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --dataset_path data/my_dataset
```

<details>
<summary><b>DDP implementation details</b></summary>

- Serialized model loading (rank 0 first, then barrier) to avoid HuggingFace cache races
- 30-minute NCCL timeout for large model initialization
- `device_map_auto=False` for DDP compatibility (no model sharding)
- Unwrapped model for validation (avoids DDP deadlock on single-rank validation)
- `DistributedSampler` with proper epoch shuffling

</details>

---

## Benchmark Results

See [docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md) for the full experiment matrix.

### VLA Pretraining

| Experiment | VLM | Head | GPUs | Train Loss | Val Loss | Throughput | Notes |
|------------|-----|------|------|-----------|----------|------------|-------|
| MLP Baseline | MockVLM | Regression | 1 × H100 | 0.1705 | 0.3899 | 3.34 batch/s | Sanity-check run |
| Qwen2-VL DDP | Qwen2-VL-2B | Regression | 3 × H100 | 0.0323 → 0.0228 | 0.0228 | ~67 samples/s | 4 epochs, AMP enabled |

### Locomotion (MJLab)

| Task | Algorithm | Envs | Reward (start → end) | Steps/s | Hardware | Iters |
|------|-----------|------|----------------------|---------|----------|-------|
| Go1 Flat | PPO (GAE) | 256 | −5.77 → −0.31 | 1,100 | 1 × H100 | 20 |

### Algorithm Verification

| Paradigm | Algorithm | Status |
|----------|-----------|--------|
| Pretrain (SL) | VLAPretrainAlgorithm | ✅ Verified (single + 3-GPU DDP) |
| SFT | SFTAlgorithm (KL reg.) | ✅ Verified (single + 2-GPU DDP) |
| GRPO | GRPOAlgorithm | ✅ Verified |
| PPO (GAE) | PPOAlgorithm | ✅ Verified |
| Locomotion PPO | PPO (MJLab Go1) | ✅ Verified (H100, 1100 steps/s) |

---

## Scripts Reference

| Script | Purpose | Docs |
|--------|---------|------|
| [`train_mjlab.py`](scripts/renforce/train_mjlab.py) | Train on MJLab environments | `--help` for all options |
| [`train_lab.py`](scripts/renforce/train_lab.py) | Train on Isaac Lab environments | Requires Isaac Sim |
| [`train_gym.py`](scripts/renforce/train_gym.py) | Train on Gymnasium/MuJoCo | Standard envs |
| [`play_mjlab.py`](scripts/renforce/play_mjlab.py) | Evaluate MJLab checkpoint | |
| [`play_lab.py`](scripts/renforce/play_lab.py) | Evaluate Isaac Lab checkpoint | Video recording |
| [`train_single_gpu.py`](scripts/vla/pretrain/train_single_gpu.py) | VLA pretraining (1 GPU) | |
| [`train_ddp.py`](scripts/vla/pretrain/train_ddp.py) | VLA pretraining (multi-GPU) | Use with `torchrun` |
| [`train_sft.py`](scripts/vla/post_train/train_sft.py) | VLA supervised fine-tuning | |
| [`train_robotwin_grpo.py`](scripts/vla/rl/train_robotwin_grpo.py) | VLA RL (GRPO/PPO) on RoboTwin | `--algo grpo/ppo` |
| [`train_vla_benchmark.py`](scripts/vla/rl/train_vla_benchmark.py) | VLA on LIBERO/ManiSkill/CALVIN/D4RL | `--benchmark libero` |

---

## Acknowledgments

- Built with alignment to [rsl_rl](https://github.com/leggedrobotics/rsl_rl) conventions
- Isaac Lab integration via [Isaac Lab](https://github.com/isaac-sim/Isaac-Lab)
- MJLab integration via [MJLab](https://github.com/mujocolab/mjlab) (MuJoCo Warp)
- VLM backbones from [HuggingFace](https://huggingface.co/) ecosystem
- Data format compatible with [LeRobot](https://github.com/huggingface/lerobot)

---

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.

## Contact

- Maintainer: Ziang Zheng — ziang_zheng@foxmail.com
