# RoboRenForce Community Testing Plan

> **Goal**: Validate algorithm effectiveness, environment integration, and convergence across tasks & datasets.
> Contributors can pick any test from the checklist, run it, and report results.

This document is **both a checklist and a tutorial**. New contributors should read sections 0 → 1 → 3 (single-process VLA RL) end-to-end before opening the tier checklist (section 2). Section 4 covers the distributed `RRF_orchestra` runner used for production RL fine-tuning.

---

## Table of Contents

0. [Prerequisites](#0-prerequisites) — system deps, Vulkan, dataset locations
1. [Quick Smoke (≤10 min)](#1-quick-smoke-10-min) — three-step verification any contributor can run
2. [Tier 1–7 Checklist](#2-tier-17-checklist) — algorithm coverage matrix (the original community-testing checklist)
3. [VLA RL — Single-Process Tutorial](#3-vla-rl--single-process-tutorial) — `scripts/vla/rl/train_*.py` walkthroughs for all 4 sim adapters
4. [VLA RL — Orchestra Distributed Tutorial](#4-vla-rl--orchestra-distributed-tutorial) — `RRF_orchestra` rollout/inference/learner roles
5. [How to Report Results](#5-how-to-report-results)
6. [Troubleshooting Matrix](#6-troubleshooting-matrix)
7. [Priority Matrix](#7-priority-matrix)

---

## 0. Prerequisites

### 0.1 System Packages

| Package | Why | Install (Ubuntu 22.04) |
|---|---|---|
| `python3.10-dev` | Cython builds | `apt-get install python3.10-dev` |
| `ffmpeg` | LeRobot video I/O | `apt-get install ffmpeg` |
| `git-lfs` | HF dataset weights | `apt-get install git-lfs && git lfs install` |
| `libvulkan1`, `vulkan-tools` | SAPIEN GPU rendering (ManiSkill, RoboTwin) | `apt-get install libvulkan1 vulkan-tools mesa-vulkan-drivers` |
| `libegl1`, `libgles2-mesa` | Headless rendering fallback | `apt-get install libegl1 libgles2-mesa` |

**Vulkan ICD discovery (NVIDIA hosts)**

SAPIEN looks for the NVIDIA ICD at `/usr/share/vulkan/icd.d/nvidia_icd.json`. The NVIDIA driver package places it at `/etc/vulkan/icd.d/nvidia_icd.json` instead, so symlink:

```bash
ln -sf /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
```

Verify:

```bash
vulkaninfo --summary | grep -A1 deviceName    # should list your NVIDIA GPU(s)
```

If the loader cannot create a Vulkan instance against the NVIDIA driver, ManiSkill rgbd-mode and RoboTwin will hang forever in their scene-init loops. Fall back to ManiSkill `--obs_mode state` until Vulkan is healthy (RoboTwin has no equivalent fallback).

### 0.2 Python Environment

The repo uses `uv` and Python 3.10:

```bash
cd RoboRenForce
uv venv --python 3.10 .venv
. .venv/bin/activate
uv pip install -e source/RoboRenForce
```

### 0.3 Per-Task Install Matrix

Install only the packages for the tier you are running:

| Tier | Packages |
|---|---|
| T1 (Gym classic) | `uv pip install -e source/tasks/RRF_gym` |
| T2 (D4RL offline) | `uv pip install -e source/tasks/RRF_d4rl` |
| T3 (MJLab loco) | `uv pip install -e source/tasks/RRF_mjlab` |
| T4 (VLA pretrain) | datasets only — no extra package |
| T5 / VLA-RL LIBERO | `uv pip install -e source/tasks/RRF_libero source/tasks/RRF_libero_vla_rl` + `pip install robosuite` + clone & install LIBERO |
| T5 / VLA-RL ManiSkill | `uv pip install -e source/tasks/RRF_maniskill source/tasks/RRF_maniskill_vla_rl` + `uv pip install mani_skill` |
| T5 / VLA-RL CALVIN | `uv pip install -e source/tasks/RRF_calvin source/tasks/RRF_calvin_vla_rl` + clone & install `calvin_env` |
| T5 / VLA-RL RoboTwin | `uv pip install -e source/tasks/RRF_robotwin source/tasks/RRF_robotwin_vla_rl` + clone & install RoboTwin assets, `mplib==0.2.1`, `toppra` |
| Orchestra | `uv pip install -e source/RRF_orchestra` |

### 0.4 Environment Variables

| Variable | When | Value |
|---|---|---|
| `HF_TOKEN` | Gated HF repos (Cosmos-Reason2, Psi0 data) | personal HF token with read scope |
| `ASSETS_PATH` | RoboTwin only | the **repo root** of your RoboTwin clone (e.g. `/path/to/RoboTwin/`), **not** `…/RoboTwin/assets/` |
| `CUDA_VISIBLE_DEVICES` | Multi-GPU host where you want to use a subset | e.g. `0,1` |
| `MUJOCO_GL` | Headless servers running MJLab | `egl` |

### 0.5 Verify the Install

```bash
python scripts/verify_environment.py
```

Expect: Python 3.10, CUDA available, all required core packages importable. Failures here mean section 0.1–0.3 is incomplete.

For SAPIEN-based sims, additionally run:

```bash
python -c "import sapien; s = sapien.Scene(); print('sapien scene ok')"
```

If this prints `sapien scene ok` cleanly, basic SAPIEN is working. (Note: this does not exercise the GPU camera — full rendering is exercised by section 1's smoke or by ManiSkill/RoboTwin themselves.)

---

## 1. Quick Smoke (≤10 min)

A three-step verification any contributor can run before tackling the tier checklist.

### Step 1 — Framework imports & unit tests (≤2 min)

```bash
pytest tests/test_rrf_models.py tests/test_grpo.py -q
```

Expect: all green. Failures here usually mean a missing `uv pip install -e source/RoboRenForce` step.

### Step 2 — Single-env step on each registered VLA-RL task (≤5 min, no learning)

```bash
# Smoke that env builds + reset + 1 step works end-to-end. No model, no GPU.
python scripts/verify_environment.py --check vla-rl-tasks
```

If you don't have all 4 sims installed, use the per-sim smoke:

```bash
# pick the sim you've installed
python -c "
import gymnasium as gym, RRF_libero_vla_rl_tasks  # registers task IDs
spec = gym.spec('LIBERO-Spatial-GRPO-v0')
env = spec.kwargs['env_cfg_entry_point'].build()
obs, info = env.reset()
print('reset ok; obs keys:', list(obs.keys()))
"
```

### Step 3 — One GRPO iteration on the simplest sim (≤5 min)

```bash
# LIBERO is the most permissive (no Vulkan needed).
./.venv/bin/python scripts/vla/rl/train_libero.py \
    --task LIBERO-Spatial-GRPO-v0 \
    --num_envs 2 \
    --max_iterations 1
```

Expect:
- `[INFO] Env built: LiberoRRFEnv (num_envs=2)`
- `[INFO] Policy built: VLAActor`
- `[INFO] Starting training for 1 iterations...`
- `[INFO] Training complete.` within ~6 min.

If this passes, the framework is healthy end-to-end. **Verified status as of last release**:

| Adapter | 1-iter smoke | Time | Notes |
|---|---|---|---|
| LIBERO | ✅ | ~335s | robosuite/MuJoCo, no Vulkan |
| ManiSkill `--obs_mode state` | ✅ | ~93s | bypasses GPU rendering |
| ManiSkill `--obs_mode rgbd` | ⚠ requires healthy Vulkan ICD | n/a | see §0.1 |
| CALVIN | ✅ | ~339s | pybullet, no Vulkan |
| RoboTwin | ⚠ requires healthy Vulkan ICD | n/a | scene-init silent retry loop without it |

If your smoke run takes >10 min for any sim that has a ✅ above, see [§6 Troubleshooting](#6-troubleshooting-matrix).

---

## 2. Tier 1–7 Checklist

The tier checklist below is the **algorithm × task convergence matrix**. Each row is a contribution-sized chunk of work. Pick a row, run it, file an issue (see §5).

### Tier 1: Classic Control — Algorithm Convergence Verification

**Hardware**: 1× GPU (any), ~10 min per run.
**Purpose**: Verify each RL algorithm converges on well-understood benchmarks.

#### T1.1 PPO on Gymnasium Tasks

| ID | Task | Env ID | Expected Reward | Max Steps | Converge By |
|---|---|---|---|---|---|
| T1.1a | CartPole-v1 | `CartPole-v1` | ≥ 475 (out of 500) | 200K | ~50K steps |
| T1.1b | Pendulum-v1 | `Pendulum-v1` | ≥ -200 | 500K | ~200K steps |
| T1.1c | LunarLander-v3 | `LunarLander-v3` | ≥ 200 | 1M | ~500K steps |
| T1.1d | BipedalWalker-v3 | `BipedalWalker-v3` | ≥ 250 | 2M | ~1M steps |

```bash
python scripts/renforce/train_gym.py --task CartPole-v1 --algo ppo \
    --num_envs 32 --max_iterations 1000 --seed 42
```

**Report**: Learning curve screenshot, final mean reward ± std (over 3 seeds: 42, 123, 456).

#### T1.2 SAC on MuJoCo Continuous Control

| ID | Task | Env ID | Expected Reward | Max Steps | Converge By |
|---|---|---|---|---|---|
| T1.2a | HalfCheetah-v5 | `HalfCheetah-v5` | ≥ 8000 | 1M | ~500K steps |
| T1.2b | Hopper-v5 | `Hopper-v5` | ≥ 3000 | 1M | ~300K steps |
| T1.2c | Walker2d-v5 | `Walker2d-v5` | ≥ 4000 | 2M | ~1M steps |
| T1.2d | Ant-v5 | `Ant-v5` | ≥ 5000 | 3M | ~1.5M steps |
| T1.2e | Humanoid-v5 | `Humanoid-v5` | ≥ 5000 | 5M | ~3M steps |

```bash
python scripts/renforce/train_gym.py --task HalfCheetah-v5 --algo sac \
    --num_envs 1 --max_iterations 1000000 --seed 42
```

#### T1.3 DSAC vs SAC

| ID | Task | Algo | Purpose |
|---|---|---|---|
| T1.3a | HalfCheetah-v5 | DSAC | Compare final perf & sample efficiency vs SAC |
| T1.3b | Hopper-v5 | DSAC | Stability comparison |
| T1.3c | Walker2d-v5 | DSAC | Medium-dim locomotion |

```bash
python scripts/renforce/train_gym.py --task HalfCheetah-v5 --algo dsac \
    --num_envs 1 --max_iterations 1000000 --seed 42
```

### Tier 2: Offline RL — D4RL Benchmark

**Hardware**: 1× GPU, ~30 min per run.

#### T2.1 IQL on D4RL Locomotion

| ID | Dataset | Quality | Expected Score | Reference |
|---|---|---|---|---|
| T2.1a | halfcheetah-medium-v2 | medium | ≥ 47.0 | IQL paper: 47.4 |
| T2.1b | halfcheetah-medium-expert-v2 | medium-expert | ≥ 86.0 | IQL paper: 86.7 |
| T2.1c | hopper-medium-v2 | medium | ≥ 60.0 | IQL paper: 66.3 |
| T2.1d | hopper-medium-expert-v2 | medium-expert | ≥ 90.0 | IQL paper: 91.5 |
| T2.1e | walker2d-medium-v2 | medium | ≥ 75.0 | IQL paper: 78.3 |
| T2.1f | walker2d-medium-expert-v2 | medium-expert | ≥ 108.0 | IQL paper: 109.6 |

```bash
python scripts/renforce/train_gym.py --task halfcheetah-medium-v2 --algo iql \
    --offline --epochs 1000 --batch_size 256 --seed 42
```

### Tier 3: Sim Locomotion — Smoothness & Robustness

**Hardware**: 1× GPU with MuJoCo Warp, ~1 hour per run.

#### T3.1 PPO Variants on Go1 Flat

| ID | Algorithm | Task | Purpose |
|---|---|---|---|
| T3.1a | PPO | `Mjlab-Velocity-Flat-Unitree-Go1` | Baseline |
| T3.1b | CAPS-PPO | same | Lipschitz smoothness effect |
| T3.1c | L2C2-PPO | same | Layer-wise contraction effect |
| T3.1d | Lips-PPO | same | 1-Lipschitz spectral norm effect |
| T3.1e | SAPG-PPO | same | Self-adaptive exploration |

```bash
python scripts/renforce/train_mjlab.py \
    --task Mjlab-Velocity-Flat-Unitree-Go1 \
    --algo ppo --num_envs 4096 --max_iterations 5000 --seed 42
```

#### T3.2 Go1 Rough Terrain

| ID | Algorithm | Task | Purpose |
|---|---|---|---|
| T3.2a | PPO | `Mjlab-Velocity-Rough-Unitree-Go1` | Baseline |
| T3.2b | CAPS-PPO | same | Smoothness on rough terrain |
| T3.2c | SAPG-PPO | same | Adaptive exploration |

#### T3.3 Humanoid G1

| ID | Algorithm | Task | Purpose |
|---|---|---|---|
| T3.3a | PPO | `Mjlab-Velocity-Flat-Unitree-G1` | Humanoid baseline |
| T3.3b | SAPG-PPO | same | Exploration on high-DoF |
| T3.3c | PPO | `Mjlab-Velocity-Rough-Unitree-G1` | Humanoid rough terrain |

### Tier 4: VLA Pretrain — Model × Action Head Convergence

**Hardware**: 1–8× GPU, ~2–8 hours per run.

#### T4.1 Psi0 Humanoid Single-Task

**Dataset**: `G1_Dex3_PickApple` (~10K frames).

| ID | VLM Backbone | Action Head | GPUs | Batch | Epochs | Key Metric |
|---|---|---|---|---|---|---|
| T4.1a | MLP Baseline | Regression | 1 | 32 | 50 | action_mse ↓ |
| T4.1b | Qwen2-VL-2B (frozen) | Regression | 1–8 | 16/gpu | 20 | action_mse ↓ |
| T4.1c | Qwen2-VL-2B (frozen) | Diffusion | 1–8 | 16/gpu | 20 | action_mse ↓ |
| T4.1d | Qwen2-VL-2B (LoRA r=16) | Regression | 1–8 | 16/gpu | 20 | action_mse ↓ |

```bash
# T4.1a
python scripts/vla/pretrain/train_single_gpu.py --mock --epochs 50 --batch_size 32

# T4.1b
torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
    --data_root <PSI0_DATA>/G1_Dex3_PickApple \
    --head regression --epochs 20 --batch_size 16

# T4.1c
torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
    --data_root <PSI0_DATA>/G1_Dex3_PickApple \
    --head diffusion --epochs 20 --batch_size 16
```

#### T4.2 Psi0 Multi-Task

**Dataset**: 10 Psi0 G1 tasks (mixture), eval on 4 held-out tasks.

| ID | Tasks | VLM | Head | Purpose |
|---|---|---|---|---|
| T4.2a | 10-task mixture | Qwen2-VL-2B | Regression | Multi-task convergence |
| T4.2b | 10-task mixture | Qwen2-VL-2B | Diffusion | Multi-task diffusion head |

#### T4.3 Cross-Backbone Comparison

| ID | VLM | Params | Action Head | Purpose |
|---|---|---|---|---|
| T4.3a | Qwen2-VL-2B | 2B | Regression | Baseline |
| T4.3b | Qwen3-VL-2B | 2B | Regression | Next-gen |
| T4.3c | OpenPI (pi0.5) | 4B | Flow matching | Native flow head |
| T4.3d | GR00T N1.7 | 3B | DiT | Native DiT head |

### Tier 5: VLA SFT & RL Fine-tuning — End-to-End Pipeline

**Hardware**: 2–8× GPU + sim env, ~4–24 hours per run.

#### T5.1 SFT KL-Regularization Ablation

**Base**: Best checkpoint from T4.1.

| ID | KL Coeff | Freeze | Purpose |
|---|---|---|---|
| T5.1a | 0.0 | backbone frozen | Baseline SFT |
| T5.1b | 0.01 | backbone frozen | Mild regularization |
| T5.1c | 0.1 | backbone frozen | Strong regularization |
| T5.1d | 0.0 | full fine-tune | No freeze, check forgetting |

```bash
python scripts/vla/post_train/train_sft.py \
    --checkpoint <PRETRAIN_CKPT> --kl_coef 0.01 \
    --epochs 10 --batch_size 16
```

#### T5.2 GRPO vs PPO on RoboTwin

**Base**: Best SFT checkpoint from T5.1. **Env**: RoboTwin PlaceEmptyCup.

| ID | RL Algo | Init | Envs | Iters | Purpose |
|---|---|---|---|---|---|
| T5.2a | GRPO | SFT ckpt | 8 | 500 | GRPO from pretrained |
| T5.2b | PPO | SFT ckpt | 8 | 500 | PPO from pretrained |
| T5.2c | GRPO | Random init | 8 | 500 | RL from scratch |
| T5.2d | PPO | Random init | 8 | 500 | RL from scratch |

```bash
python scripts/vla/rl/train_robotwin.py \
    --task RoboTwin-PlaceCup-GRPO-v0 \
    --checkpoint <SFT_CKPT> --num_envs 8 --max_iterations 500
```

#### T5.3 Full Pipeline Across Sims

| ID | Sim | Task ID | Pipeline |
|---|---|---|---|
| T5.3a | RoboTwin | `RoboTwin-PlaceCup-GRPO-v0` | Pretrain → SFT → GRPO |
| T5.3b | LIBERO | `LIBERO-Spatial-GRPO-v0` | Pretrain → SFT → GRPO |
| T5.3c | ManiSkill | `ManiSkill-PickCube-GRPO-v0` | Pretrain → SFT → GRPO |
| T5.3d | CALVIN | `CALVIN-D-GRPO-v0` | Pretrain → SFT → GRPO |

### Tier 6: Imitation Learning & Model-Based

#### T6.1 GAIL + PPO

| ID | Task | Expert Source | Purpose |
|---|---|---|---|
| T6.1a | Go1 Flat (MJLab) | Trained PPO policy | GAIL from expert demos |
| T6.1b | HalfCheetah-v5 | D4RL expert data | GAIL on standard benchmark |

#### T6.2 MBPO on MuJoCo

| ID | Task | Purpose |
|---|---|---|
| T6.2a | HalfCheetah-v5 | MBPO sample efficiency |
| T6.2b | Hopper-v5 | MBPO unstable dynamics |

#### T6.3 Distillation

| ID | Teacher | Student | Task |
|---|---|---|---|
| T6.3a | Large PPO (256-256) | Small MLP (64-64) | Go1 Flat |
| T6.3b | Qwen2-VL actor | MLP actor | PickApple |

### Tier 7: Scalability & Robustness

#### T7.1 DDP Scaling Efficiency

| ID | GPUs | Task |
|---|---|---|
| T7.1a | 1 | VLA Pretrain (PickApple) |
| T7.1b | 2 | same |
| T7.1c | 4 | same |
| T7.1d | 8 | same |

#### T7.2 Seed Robustness

| ID | Task | Algo | Seeds |
|---|---|---|---|
| T7.2a | HalfCheetah-v5 | SAC | 42,123,456,789,0 |
| T7.2b | Go1 Flat | PPO | 42,123,456,789,0 |
| T7.2c | PickApple Pretrain | VLA | 42,123,456 |

#### T7.3 Mixed Precision

| ID | Task | Precision |
|---|---|---|
| T7.3a | VLA Pretrain | fp32 |
| T7.3b | VLA Pretrain | bf16 |
| T7.3c | PPO Go1 | fp32 vs bf16 |

#### T7.4 Orchestra Distributed RL — Throughput vs Single-Process

| ID | Topology | Sim | Purpose |
|---|---|---|---|
| T7.4a | 1 env worker / 1 inference worker | LIBERO-Spatial | Validate orchestra wiring matches single-process reward curve |
| T7.4b | 4 env workers / 1 inference worker | LIBERO-Spatial | Throughput scaling |
| T7.4c | 8 env workers / 1 inference worker | RoboTwin-PlaceCup | Bottleneck identification (env vs inference vs learner) |

See §4.5 for the topology spec and §4.6 for current status.

---

## 3. VLA RL — Single-Process Tutorial

The single-process VLA RL runner is what every `scripts/vla/rl/train_<sim>.py` invokes. Read this before running anything in §2 Tier 5 or §4 Orchestra. The picture:

```
       train_<sim>.py
           │
           ▼  (parses CLI, gym.spec lookup)
       _common.py::run(args)
           │
           ├─── env_cfg.build()   ──►  <Sim>RRFEnv  (vectorized, num_envs)
           │
           ├─── runner_cfg.build_policy()  ──►  VLAActor (Qwen2-VL + head)
           │
           └─── runner_cfg.class_type(env, policy, …).learn(num_iterations)
                     │
                     └── GRPO / PPO algorithm consumes rollouts
```

### 3.1 The Three Cfg Objects You Need to Know

For every registered VLA-RL task ID (e.g. `LIBERO-Spatial-GRPO-v0`), `gym.spec(task_id).kwargs` exposes:

| Key | Type | Defines |
|---|---|---|
| `env_cfg_entry_point` | dataclass instance | Env construction (sim name, num_envs, image_size, obs_mode, reward shaping…) |
| `RoboRenForce_entry_point` | runner cfg instance | Algorithm + policy + iteration count |

You override these from CLI flags or by editing the cfg files under `source/tasks/RRF_<sim>_vla_rl_tasks/<task>/env_cfg.py`. The standard CLI flags (in `scripts/vla/rl/_common.py`):

| Flag | Effect |
|---|---|
| `--task <id>` | Select registered task ID |
| `--num_envs N` | Override `env_cfg.num_envs` |
| `--device cuda:0` | Compute device |
| `--max_iterations N` | Number of RL iterations |
| `--seed N` | Reproducibility |
| `--logdir PATH` | Override default `logs/RFRL/<task>/<ts>` |
| `--checkpoint PATH` | Load pretrained / SFT checkpoint into the policy |
| `--obs_mode {state\|rgbd}` | ManiSkill: bypass GPU rendering by passing `state` |

### 3.2 ManiSkill PickCube (the simplest)

Files:
- env adapter: `source/tasks/RRF_maniskill/RRF_maniskill_tasks/envs/maniskill_env.py`
- env cfg: `source/tasks/RRF_maniskill_vla_rl/RRF_maniskill_vla_rl_tasks/pick_cube/env_cfg.py`
- runner cfg: `…/pick_cube/agents_grpo.py` (`ManiSkillPickCubeGRPOCfg`)
- training script: `scripts/vla/rl/train_maniskill.py`

State-mode 1-iter smoke (works without Vulkan):

```bash
./.venv/bin/python scripts/vla/rl/train_maniskill.py \
    --task ManiSkill-PickCube-GRPO-v0 \
    --num_envs 2 --max_iterations 1 --obs_mode state
```

Full rgbd training (requires Vulkan, see §0.1):

```bash
./.venv/bin/python scripts/vla/rl/train_maniskill.py \
    --task ManiSkill-PickCube-GRPO-v0 \
    --num_envs 16 --max_iterations 500 --seed 42
```

### 3.3 LIBERO Spatial

Files:
- env adapter: `source/tasks/RRF_libero/RRF_libero_tasks/envs/libero_env.py`
- env cfg: `source/tasks/RRF_libero_vla_rl/RRF_libero_vla_rl_tasks/spatial/env_cfg.py`
- runner cfg: `…/spatial/agents_grpo.py`
- training script: `scripts/vla/rl/train_libero.py`

```bash
./.venv/bin/python scripts/vla/rl/train_libero.py \
    --task LIBERO-Spatial-GRPO-v0 \
    --num_envs 4 --max_iterations 500 --seed 42
```

LIBERO uses robosuite/MuJoCo for offscreen rendering — no Vulkan required, but it does scale linearly across `num_envs` (no GPU vectorization). Cap at ~8 per host.

### 3.4 CALVIN D-Split

Files:
- env adapter: `source/tasks/RRF_calvin/RRF_calvin_tasks/envs/calvin_env.py`
- env cfg: `source/tasks/RRF_calvin_vla_rl/RRF_calvin_vla_rl_tasks/d_split/env_cfg.py`
- runner cfg: `…/d_split/agents_grpo.py`
- training script: `scripts/vla/rl/train_calvin.py`

```bash
./.venv/bin/python scripts/vla/rl/train_calvin.py \
    --task CALVIN-D-GRPO-v0 \
    --num_envs 4 --max_iterations 500 --seed 42
```

If you have a CALVIN dataset release, set `cfg["dataset_path"]` in the env cfg to load the recorded scene config; without it, the adapter falls back to the upstream `config_data_collection` Hydra config (good for smoke / CI but not for reproducing benchmark numbers).

### 3.5 RoboTwin PlaceCup

Files:
- env adapter: `source/tasks/RRF_robotwin/RRF_robotwin_tasks/envs/robotwin_env.py`
- env cfg: `source/tasks/RRF_robotwin_vla_rl/RRF_robotwin_vla_rl_tasks/place_cup/env_cfg.py`
- runner cfg: `…/place_cup/agents_grpo.py` and `agents_ppo.py`
- training script: `scripts/vla/rl/train_robotwin.py`

```bash
ASSETS_PATH=/path/to/RoboTwin/ \
./.venv/bin/python scripts/vla/rl/train_robotwin.py \
    --task RoboTwin-PlaceCup-GRPO-v0 \
    --num_envs 4 --max_iterations 500 --seed 42
```

**Gotchas**:
- `ASSETS_PATH` must point to the RoboTwin **repo root** (the dir containing `assets/`), not `…/RoboTwin/assets/`.
- `mplib` must be ≥ 0.2.1 (older 0.1.x is missing `sapien_utils`). Upgrade with `uv pip install 'mplib>=0.2.1'`.
- RoboTwin scene init **silently retries on every Vulkan error** in `RoboTwin/robotwin/envs/vector_env.py::setup_task`. If you don't see progress in the first 60s of `[INFO] Env built…`, your Vulkan ICD is broken — see §0.1.

### 3.6 Adding a New VLA-RL Task

1. Pick or implement the env adapter under `source/tasks/RRF_<sim>/<sim>_tasks/envs/<sim>_env.py` (must implement `EmbodiedEnv`).
2. Create `source/tasks/RRF_<sim>_vla_rl/RRF_<sim>_vla_rl_tasks/<my_task>/env_cfg.py` with a `@dataclass MyTaskEnvCfg` whose `build()` returns the env instance.
3. Create `…/<my_task>/agents_grpo.py` with a `MyTaskGRPOCfg` that subclasses `VLAGRPORunnerCfg`. Implement `build_policy()` to return a `VLAActor`.
4. Register the task ID in `…/RRF_<sim>_vla_rl_tasks/__init__.py`:
   ```python
   register_<sim>_task(
       "<MySim>-MyTask-GRPO-v0",
       MyTaskEnvCfg(),
       MyTaskGRPOCfg(),
   )
   ```
5. Add a smoke entry to §1 Step 3 of this doc.

---

## 4. VLA RL — Orchestra Distributed Tutorial

The single-process runner above couples three roles in one process: the env steps, the policy forward pass, and the optimizer update. For real scale (large VLM + many parallel envs + multi-GPU learner), you want them in **separate processes with explicit channels**. That's `RRF_orchestra`.

### 4.1 Why Orchestra

| Bottleneck (single-process) | Orchestra fix |
|---|---|
| Env step blocks GPU forward | Move env to separate process(es); learner GPU never idle |
| One inference batch per env | Inference worker batches obs from N env workers |
| Weight update preempts rollout | Learner runs in dedicated process; weights pushed asynchronously |
| Sims with conflicting Python deps | Each sim runs in its own process — import isolation |

This is the same role-decomposition that VeRL / RLinf / OpenRLHF / NeMo-RL adopt for VLA + LLM RL.

### 4.2 Topology

`TopologyCfg` (in `source/RRF_orchestra/RRF_orchestra/orchestrator/topology.py`) wires:

```
┌──────────────┐  obs  ┌──────────────────┐ action[i] ┌──────────────┐
│ EnvWorker[0] ├──────►│ InferenceWorker  ├──────────►│ EnvWorker[i] │
│     ⋮        │       │  (batches N→1)   │           │      ⋮       │
│ EnvWorker[N] ├──────►│                  │           │ EnvWorker[N] │
└─────┬────────┘       └────────┬─────────┘           └──────────────┘
      │ traj                    │ weight_request
      ▼                         ▼
┌─────────────────────────────────────────────┐
│  OrchestraVLARunner  (learner process)      │
│  - holds policy + optimizer                 │
│  - consumes Trajectory messages             │
│  - broadcasts weights to inference worker   │
└─────────────────────────────────────────────┘
```

Channels (defaults, tunable):

| Channel | Direction | Capacity |
|---|---|---|
| `obs_ch` | N env → 1 inference | `4 × N` |
| `action_chs[i]` | 1 inference → 1 env | 4 per env |
| `traj_ch` | N env → 1 learner | `2 × N` |
| `weight_ch` | 1 learner → 1 inference | 2 (newest wins) |
| `ctrl_in_chs[i]` | supervisor → worker | 8 per worker |
| `ctrl_out_ch` | workers → supervisor | `8 × M` shared |

### 4.3 Hello-World

The smallest end-to-end test that exercises every channel:

```bash
python -m RRF_orchestra.examples.hello_world_orchestra
```

Source: `source/RRF_orchestra/RRF_orchestra/examples/hello_world_orchestra.py`. It builds:

- 2 fake env workers that emit `Trajectory` messages with constant rewards.
- 1 inference worker with a noop `nn.Linear(4, 7)` policy.
- A learner that runs 5 iterations of a fake algorithm (just bumps a counter and steps the optimizer).

Expected output ends with `finished — collected 5 iteration(s)`.

This validates: process spawn → channels created → workers reach setup → obs flow → action flow → traj flow → learner update → weight broadcast. If hello-world fails, no real task will work.

### 4.4 Recipe: Wrapping a Real Sim Env into an Env Worker

Take `LiberoRRFEnv` (or any `EmbodiedEnv`) and turn it into a `BaseEnvWorker` subclass.

```python
# source/tasks/RRF_libero_vla_rl/RRF_libero_vla_rl_tasks/spatial/env_worker.py
from RoboRenForce.utils.configclass import configclass
from RRF_orchestra.workers.env_worker import BaseEnvWorker, BaseEnvWorkerCfg
from RRF_orchestra.protocol.messages import ObsBatch, Trajectory
from RRF_orchestra.protocol.shared_tensor import SharedTensorRef
import torch, time

from RRF_libero_tasks.envs.libero_env import LiberoRRFEnv

class LiberoSpatialEnvWorker(BaseEnvWorker):
    def setup(self) -> None:
        # build the actual sim inside the worker process
        self.env = LiberoRRFEnv(
            cfg={"task_suite_name": "libero_spatial",
                 "image_size": (224, 224),
                 "max_episode_steps": 300},
            num_envs=self.cfg.num_envs_per_worker,
            device="cpu",
        )

    def reset_envs(self, env_ids):
        obs, _ = self.env.reset()
        return self._wrap_obs(obs, env_ids, step=0)

    def step_envs(self, action):
        # action is a Tensor[B, action_dim]
        obs, reward, done, info = self.env.step(action.tensor)
        return self._wrap_obs(obs, list(range(action.tensor.shape[0])), step=...), reward, done, []

    def _wrap_obs(self, obs, env_ids, step):
        return ObsBatch(
            worker_ids=[self.cfg.worker_id]*len(env_ids),
            env_ids=env_ids,
            step_ids=[step]*len(env_ids),
            states=SharedTensorRef.from_tensor(obs["states"].cpu()),
            timestamp=time.time(),
        )

@configclass
class LiberoSpatialEnvWorkerCfg(BaseEnvWorkerCfg):
    class_type: type = LiberoSpatialEnvWorker
    name: str = "libero_spatial_env"
```

Then build a topology + runner:

```python
# scripts/vla/rl/orchestra/train_libero_orchestra.py
from RRF_orchestra.orchestrator.orchestra_runner import (
    OrchestraVLARunner, OrchestraVLARunnerCfg,
)
from RRF_orchestra.orchestrator.topology import TopologyCfg
from RRF_orchestra.workers.inference_worker import InferenceWorkerCfg

from RRF_libero_vla_rl_tasks.spatial.env_worker import LiberoSpatialEnvWorkerCfg
from RRF_libero_vla_rl_tasks.spatial.agents_grpo import LiberoSpatialGRPOCfg

def main():
    grpo_cfg = LiberoSpatialGRPOCfg()
    topology = TopologyCfg(
        num_env_workers=4,
        env_worker_cfg=LiberoSpatialEnvWorkerCfg(num_envs_per_worker=1),
        inference_worker_cfg=InferenceWorkerCfg(
            policy_factory=grpo_cfg.build_policy,
        ),
    )
    runner_cfg = OrchestraVLARunnerCfg(
        topology=topology,
        learner_policy_factory=grpo_cfg.build_policy,
        optimizer_factory=lambda p: torch.optim.AdamW(p.parameters(), lr=1e-5),
        algorithm_factory=lambda: AlgorithmAdapter(grpo_cfg.build_algorithm()),
        max_iterations=500,
        batch_size=4,            # min trajs per learner update
        weight_sync_every=1,
        collect_timeout_s=60.0,
    )
    OrchestraVLARunner(runner_cfg).learn(
        on_iter_end=lambda it, m: print(f"[iter {it}] {m}")
    )
```

Replace `LiberoSpatial*` with `RoboTwinPlaceCup*`, `ManiSkillPickCube*`, or `CalvinDSplit*` to hit the other sims. The pattern is identical.

### 4.5 Multi-GPU Learner

The learner is a single process today; for multi-GPU update use `torchrun` around the orchestra runner. Each learner replica runs its own `OrchestraVLARunner` with `weight_sync_every=1` so all replicas push to their own inference worker. **Cross-replica gradient sync** is your existing DDP — wrap `policy_factory()` with `torch.nn.parallel.DistributedDataParallel`.

This is **two orthogonal parallelism axes**:

| Axis | Purpose | Knob |
|---|---|---|
| Orchestra (rollout / inference / learner) | Hide env latency, batch inference | `num_env_workers`, `batch_size` |
| DDP (across replicas) | Scale learner gradient | `torchrun --nproc_per_node` |

### 4.6 Current Status & Roadmap

| Capability | Status |
|---|---|
| Hello-world end-to-end (fake env + noop policy) | ✅ in `examples/hello_world_orchestra.py` |
| `BaseEnvWorker` for the 4 sim adapters | ⚠ not yet — see §4.4 recipe |
| Multi-GPU learner (DDP around orchestra) | ⚠ wired but no smoke yet |
| YAML-driven topology cfg | ❌ all configs are Python configclass |

Contributions writing concrete env workers (one per sim) are exactly what §7.4 of the tier checklist needs.

---

## 5. How to Report Results

### Required Format

```markdown
## Test ID: T1.2a — SAC on HalfCheetah-v5

**Environment**:
- GPU: NVIDIA RTX 4090 24GB
- CUDA: 12.x
- Python: 3.10.x
- PyTorch: 2.x

**Command**:
```
python scripts/renforce/train_gym.py --task HalfCheetah-v5 --algo sac ...
```

**Results**:
| Seed | Final Reward (mean ± std) | Steps to Converge | Wall Time |
|------|--------------------------|-------------------|-----------|
| 42   | 8234 ± 312               | 480K              | 45 min    |
| 123  | 8102 ± 298               | 520K              | 46 min    |
| 456  | 8310 ± 287               | 460K              | 44 min    |

**Learning Curve**: [attach image]

**Issues/Notes**: (any bugs, unexpected behavior, or suggestions)
```

### Issue Labels

- `benchmark-result` — successful test report
- `convergence-issue` — algorithm did not converge as expected
- `performance-gap` — result significantly below reference
- `bug` — crash, error, or incorrect behavior

---

## 6. Troubleshooting Matrix

| Symptom | Likely cause | Fix |
|---|---|---|
| RoboTwin: `setup_task` runs forever, log silent past `[INFO] Env built…` | Vulkan ICD missing / NVIDIA ICD not in `/usr/share/vulkan/icd.d/` | Install `libvulkan1`; symlink NVIDIA ICD per §0.1 |
| RoboTwin: `TypeError: expected str, bytes or os.PathLike, not NoneType` in `os.path.join` | `ASSETS_PATH` env var unset | `export ASSETS_PATH=/path/to/RoboTwin/` (repo root, **not** `.../assets/`) |
| RoboTwin: `ModuleNotFoundError: mplib.sapien_utils` | mplib too old | `uv pip install 'mplib>=0.2.1'` |
| ManiSkill: hangs on `_sapien_gpu_setup_sensors` | Vulkan loader installed but ICD not discoverable, or no GPU display server | Symlink ICD per §0.1; or fall back to `--obs_mode state` |
| ManiSkill: shape mismatch `1578 vs 1561` after env build | State-mode obs dim ≠ configured `state_dim` | Already fixed in `maniskill_env.py` (clip/pad). Run `git pull` and rebuild venv. |
| ManiSkill: device mismatch on `done_mask` (cuda vs cpu) | terminated/truncated returned as cpu tensor | Already fixed in `maniskill_env.py`. Run `git pull`. |
| CALVIN: `NameNotFound: 'CALVIN-DSplit-GRPO'` | Wrong task ID | Use `CALVIN-D-GRPO-v0` (note the **-v0**) |
| CALVIN: `TypeError: got unexpected keyword argument '_target_'` | Hydra cfg leaks instantiate marker | Already fixed in `calvin_env.py` (strip `_target_`). Run `git pull`. |
| CALVIN: `AttributeError: 'dict' has no attribute 'width'` | OmegaConf collapsed to plain dict | Already fixed (keep DictConfig). Run `git pull`. |
| CALVIN: `InterpolationResolutionError: cameras: ${cameras}` | Hydra interpolation not resolved | Already fixed (`OmegaConf.resolve(env_cfg)`). Run `git pull`. |
| CALVIN: `AssertionError: gripper_action not in (-1, 1)` | Continuous policy output, CALVIN expects discrete | Already fixed (threshold at 0). Run `git pull`. |
| LIBERO: `Benchmark constructor takes string` failure | LIBERO `Benchmark` API changed | Already fixed (use `get_benchmark` / `get_libero_path`). Run `git pull`. |
| Orchestra: hello-world hangs at `start_all` | Worker process crashed at `setup()` | Inspect `ctrl_out_ch` events; check for `import` failures in worker subprocess |
| Orchestra: `collect_timeout_s` exceeded | Env worker too slow / inference batch too big | Decrease `inference_batch_size`; increase `collect_timeout_s`; lower `num_envs_per_worker` |

### Diagnosing a Hung Sim Smoke

When a smoke run is past its expected wall-clock and the log is silent, dump the Python stack of the stuck PID:

```bash
pip install py-spy
py-spy dump --pid <pid>
```

`_sapien_gpu_setup_sensors` in the dump → Vulkan ICD problem (§0.1).
`setup_task` retry loop → same root cause for RoboTwin.
`hydra.compose` → Hydra config / interpolation issue.
`update_render` from synthetic SAPIEN test → missing display backend, try `MUJOCO_GL=egl` or run inside a host with a working DRI device.

---

## 7. Priority Matrix

| Priority | Tests | Difficulty | Time | GPU |
|---|---|---|---|---|
| 🔴 P0 (Critical) | T1.1, T1.2, T1.3, §1 smoke | Easy | 10–30 min | 1× any |
| 🟠 P1 (High) | T2.1, T3.1, T3.2 | Medium | 30–60 min | 1× GPU |
| 🟡 P2 (Medium) | T4.1, T5.1, T5.2 | Medium | 2–8 hrs | 1–8× GPU |
| 🟢 P3 (Nice-to-have) | T4.2, T4.3, T5.3, T6.x | Hard | 4–24 hrs | 2–8× GPU |
| 🔵 P4 (Advanced) | T7.1–T7.3 | Medium | 1–4 hrs | 1–8× GPU |
| 🟣 P5 (Orchestra) | T7.4, §4.4 env workers | Hard | 4–24 hrs | 2–8× GPU |

When in doubt, start with **P0 §1 Quick Smoke**, file a `benchmark-result` issue, and pick a P1 row that matches your hardware.
