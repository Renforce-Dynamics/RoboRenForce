# RoboRenForce Community Testing Plan

> **Goal**: Validate algorithm effectiveness and convergence across tasks & datasets.
> Contributors can pick any test from the checklist, run it, and report results.

---

## Quick Start

```bash
git clone https://github.com/your-org/RoboRenForce.git && cd RoboRenForce
pip install -e "source/RoboRenForce"
pip install -e "source/tasks/RRF_gym"       # Tier 1 tests
pip install -e "source/tasks/RRF_d4rl"      # Tier 2 tests
pip install -e "source/tasks/RRF_mjlab"     # Tier 3 tests
# pip install -e "source/tasks/RRF_libero"  # Tier 4 (optional)
```

---

## Tier 1: Classic Control — Algorithm Convergence Verification

**Hardware**: 1× GPU (any), ~10 min per run
**Purpose**: Verify each RL algorithm converges on well-understood benchmarks

### T1.1 PPO on Gymnasium Tasks

| ID | Task | Env ID | Expected Reward | Max Steps | Converge By |
|----|------|--------|----------------|-----------|-------------|
| T1.1a | CartPole-v1 | `CartPole-v1` | ≥ 475 (out of 500) | 200K | ~50K steps |
| T1.1b | Pendulum-v1 | `Pendulum-v1` | ≥ -200 | 500K | ~200K steps |
| T1.1c | LunarLander-v3 | `LunarLander-v3` | ≥ 200 | 1M | ~500K steps |
| T1.1d | BipedalWalker-v3 | `BipedalWalker-v3` | ≥ 250 | 2M | ~1M steps |

```bash
python scripts/renforce/train_gym.py --task CartPole-v1 --algo ppo \
    --num_envs 32 --max_iterations 1000 --seed 42
```

**Report**: Learning curve screenshot, final mean reward ± std (over 3 seeds: 42, 123, 456).

### T1.2 SAC on MuJoCo Continuous Control

| ID | Task | Env ID | Expected Reward | Max Steps | Converge By |
|----|------|--------|----------------|-----------|-------------|
| T1.2a | HalfCheetah-v5 | `HalfCheetah-v5` | ≥ 8000 | 1M | ~500K steps |
| T1.2b | Hopper-v5 | `Hopper-v5` | ≥ 3000 | 1M | ~300K steps |
| T1.2c | Walker2d-v5 | `Walker2d-v5` | ≥ 4000 | 2M | ~1M steps |
| T1.2d | Ant-v5 | `Ant-v5` | ≥ 5000 | 3M | ~1.5M steps |
| T1.2e | Humanoid-v5 | `Humanoid-v5` | ≥ 5000 | 5M | ~3M steps |

```bash
python scripts/renforce/train_gym.py --task HalfCheetah-v5 --algo sac \
    --num_envs 1 --max_iterations 1000000 --seed 42
```

**Report**: Learning curve, final mean reward ± std (3 seeds).

### T1.3 DSAC (Distributional SAC) vs SAC Head-to-Head

| ID | Task | Algo | Purpose |
|----|------|------|---------|
| T1.3a | HalfCheetah-v5 | DSAC | Compare final perf & sample efficiency vs SAC |
| T1.3b | Hopper-v5 | DSAC | Stability comparison (Hopper is unstable) |
| T1.3c | Walker2d-v5 | DSAC | Compare on medium-dim locomotion |

```bash
python scripts/renforce/train_gym.py --task HalfCheetah-v5 --algo dsac \
    --num_envs 1 --max_iterations 1000000 --seed 42
```

**Report**: Overlaid learning curves (SAC vs DSAC), wall-clock time, final reward.

---

## Tier 2: Offline RL — D4RL Benchmark

**Hardware**: 1× GPU, ~30 min per run
**Purpose**: Validate IQL on standard offline RL benchmarks

### T2.1 IQL on D4RL Locomotion

| ID | Dataset | Quality | Expected Score | Reference |
|----|---------|---------|---------------|-----------|
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

**Report**: Normalized D4RL score (3 seeds), comparison table vs IQL paper.

---

## Tier 3: Sim Locomotion — Smoothness & Robustness Algorithms

**Hardware**: 1× GPU with MuJoCo Warp, ~1 hour per run
**Purpose**: Validate smooth policy variants on quadruped/humanoid locomotion

### T3.1 PPO vs Smooth PPO Variants on Go1 Flat

| ID | Algorithm | Task | Purpose |
|----|-----------|------|---------|
| T3.1a | PPO (baseline) | Mjlab-Velocity-Flat-Unitree-Go1 | Baseline reward & gait quality |
| T3.1b | CAPS-PPO | Same | Lipschitz smoothness constraint effect |
| T3.1c | L2C2-PPO | Same | Layer-wise contraction effect |
| T3.1d | Lips-PPO | Same | 1-Lipschitz via spectral norm effect |
| T3.1e | SAPG-PPO | Same | Self-adaptive exploration effect |

```bash
python scripts/renforce/train_mjlab.py \
    --task Mjlab-Velocity-Flat-Unitree-Go1 \
    --algo ppo --num_envs 4096 --max_iterations 5000 --seed 42
```

**Report**: (1) Reward curve, (2) velocity tracking error, (3) action smoothness (∆a std), (4) sim-to-real readiness score.

### T3.2 Go1 Rough Terrain Generalization

| ID | Algorithm | Task | Purpose |
|----|-----------|------|---------|
| T3.2a | PPO | Mjlab-Velocity-Rough-Unitree-Go1 | Baseline on rough terrain |
| T3.2b | CAPS-PPO | Same | Does smoothness help on rough terrain? |
| T3.2c | SAPG-PPO | Same | Does adaptive exploration help? |

**Report**: Same metrics as T3.1, plus terrain robustness analysis (success rate on terrains of varying difficulty).

### T3.3 Humanoid G1 Locomotion

| ID | Algorithm | Task | Purpose |
|----|-----------|------|---------|
| T3.3a | PPO | Mjlab-Velocity-Flat-Unitree-G1 | Humanoid baseline |
| T3.3b | SAPG-PPO | Same | Exploration benefit on high-DoF |
| T3.3c | PPO | Mjlab-Velocity-Rough-Unitree-G1 | Humanoid rough terrain |

**Report**: Reward curve, balance stability, joint velocity profiles.

---

## Tier 4: VLA Pretrain — Model × Action Head Convergence

**Hardware**: 1–8× GPU (model-dependent), ~2–8 hours per run
**Purpose**: Validate VLA pretraining convergence on robotic manipulation data

### T4.1 Psi0 Humanoid — Single-Task Pretrain

**Dataset**: G1_Dex3_PickApple (~10K frames)

| ID | VLM Backbone | Action Head | GPUs | Batch | Epochs | Key Metric |
|----|-------------|-------------|------|-------|--------|------------|
| T4.1a | MLP Baseline (no VLM) | Regression | 1 | 32 | 50 | action_mse ↓ |
| T4.1b | Qwen2-VL-2B (frozen) | Regression | 1–8 | 16/gpu | 20 | action_mse ↓ |
| T4.1c | Qwen2-VL-2B (frozen) | Diffusion | 1–8 | 16/gpu | 20 | action_mse ↓ |
| T4.1d | Qwen2-VL-2B (LoRA r=16) | Regression | 1–8 | 16/gpu | 20 | action_mse ↓ |

```bash
# T4.1a: MLP Baseline
python scripts/vla/pretrain/train_single_gpu.py --mock --epochs 50 --batch_size 32

# T4.1b: Qwen2-VL + Regression
torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
    --data_root <PSI0_DATA>/G1_Dex3_PickApple \
    --head regression --epochs 20 --batch_size 16

# T4.1c: Qwen2-VL + Diffusion
torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
    --data_root <PSI0_DATA>/G1_Dex3_PickApple \
    --head diffusion --epochs 20 --batch_size 16
```

**Report**: Train/val loss curves, action MSE per joint, total train time.

### T4.2 Psi0 Humanoid — Multi-Task Pretrain

**Dataset**: 10 Psi0 G1 tasks (mixture), eval on 4 held-out tasks

| ID | Tasks | VLM | Head | Purpose |
|----|-------|-----|------|---------|
| T4.2a | 10-task mixture | Qwen2-VL-2B | Regression | Multi-task convergence |
| T4.2b | 10-task mixture | Qwen2-VL-2B | Diffusion | Multi-task diffusion head |

**Report**: Per-task val loss, held-out task generalization error, confusion analysis.

### T4.3 Cross-Backbone Comparison (Advanced)

| ID | VLM | Params | Action Head | Purpose |
|----|-----|--------|-------------|---------|
| T4.3a | Qwen2-VL-2B | 2B | Regression | Baseline backbone |
| T4.3b | Qwen3-VL-2B | 2B | Regression | Next-gen comparison |
| T4.3c | OpenPI (pi0.5) | 4B | Flow matching | Flow-match native head |
| T4.3d | GR00T N1.7 | 3B | DiT | DiT native head |

**Report**: Val loss, action MSE, inference latency (ms/step), VRAM usage.

---

## Tier 5: VLA SFT & RL Fine-tuning — End-to-End Pipeline

**Hardware**: 2–8× GPU + sim environment, ~4–24 hours per run
**Purpose**: Validate the full Pretrain → SFT → RL pipeline

### T5.1 SFT Ablation (KL Regularization)

**Base**: Best checkpoint from T4.1

| ID | KL Coeff | Freeze | Purpose |
|----|----------|--------|---------|
| T5.1a | 0.0 (no KL) | backbone frozen | Baseline SFT |
| T5.1b | 0.01 | backbone frozen | Mild regularization |
| T5.1c | 0.1 | backbone frozen | Strong regularization |
| T5.1d | 0.0 | full fine-tune | No freeze, check catastrophic forgetting |

```bash
python scripts/vla/post_train/train_sft.py \
    --checkpoint <PRETRAIN_CKPT> --kl_coef 0.01 \
    --epochs 10 --batch_size 16
```

**Report**: Val loss, action MSE, KL divergence vs pretrained model.

### T5.2 GRPO vs PPO on RoboTwin

**Base**: Best SFT checkpoint from T5.1
**Env**: RoboTwin PlaceEmptyCup (SAPIEN3)

| ID | RL Algo | Init | Envs | Iters | Purpose |
|----|---------|------|------|-------|---------|
| T5.2a | GRPO | SFT checkpoint | 8 | 500 | GRPO from pretrained |
| T5.2b | PPO | SFT checkpoint | 8 | 500 | PPO from pretrained |
| T5.2c | GRPO | Random init | 8 | 500 | Ablation: RL from scratch |
| T5.2d | PPO | Random init | 8 | 500 | Ablation: RL from scratch |

```bash
python scripts/vla/rl/train_robotwin_grpo.py \
    --task place_empty_cup --algo grpo \
    --checkpoint <SFT_CKPT> --num_envs 8 --max_iterations 500
```

**Report**: Success rate curve, mean return, sample efficiency (steps to 50% SR).

### T5.3 Full Pipeline on Multiple RoboTwin Tasks

| ID | Task | Pipeline | Purpose |
|----|------|----------|---------|
| T5.3a | PlaceEmptyCup | Pretrain → SFT → GRPO | Single-task full pipeline |
| T5.3b | PickApple | Pretrain → SFT → GRPO | Different manipulation type |
| T5.3c | StackBlocks | Pretrain → SFT → GRPO | Long-horizon task |
| T5.3d | OpenDrawer | Pretrain → SFT → GRPO | Articulated object task |

**Report**: Per-stage metrics, end-to-end success rate improvement.

---

## Tier 6: Imitation Learning & Model-Based RL

**Hardware**: 1–4× GPU, ~2–8 hours
**Purpose**: Validate specialized algorithm families

### T6.1 GAIL + PPO on Locomotion

| ID | Task | Expert Source | Purpose |
|----|------|--------------|---------|
| T6.1a | Go1 Flat (MJLab) | Trained PPO policy | GAIL learns from expert demos |
| T6.1b | HalfCheetah-v5 (Gym) | D4RL expert data | GAIL on standard benchmark |

**Report**: GAIL reward curve vs PPO-only, discriminator accuracy, policy similarity to expert.

### T6.2 MBPO (Model-Based) on MuJoCo

| ID | Task | Purpose |
|----|------|---------|
| T6.2a | HalfCheetah-v5 | MBPO sample efficiency vs SAC |
| T6.2b | Hopper-v5 | MBPO on unstable dynamics |

**Report**: Sample efficiency curve (MBPO vs SAC vs PPO at same wall-clock), dynamics model prediction error.

### T6.3 Distillation: Teacher → Student

| ID | Teacher | Student | Task | Purpose |
|----|---------|---------|------|---------|
| T6.3a | Large PPO (256-256) | Small MLP (64-64) | Go1 Flat | Policy compression ratio |
| T6.3b | Qwen2-VL actor | MLP actor | PickApple | VLA → lightweight distillation |

**Report**: Student vs teacher reward, param count reduction, inference speedup.

---

## Tier 7: Scalability & Robustness

**Hardware**: 1–8× GPU
**Purpose**: Framework-level reliability tests

### T7.1 DDP Scaling Efficiency

| ID | GPUs | Task | Purpose |
|----|------|------|---------|
| T7.1a | 1 GPU | VLA Pretrain (PickApple) | Baseline throughput |
| T7.1b | 2 GPUs | Same | 2-GPU scaling factor |
| T7.1c | 4 GPUs | Same | 4-GPU scaling factor |
| T7.1d | 8 GPUs | Same | 8-GPU scaling factor |

**Report**: Throughput (samples/sec), scaling efficiency (%), GPU utilization, memory usage.

### T7.2 Seed Robustness

| ID | Task | Algo | Seeds | Purpose |
|----|------|------|-------|---------|
| T7.2a | HalfCheetah-v5 | SAC | 42,123,456,789,0 | Variance across 5 seeds |
| T7.2b | Go1 Flat | PPO | 42,123,456,789,0 | Variance across 5 seeds |
| T7.2c | PickApple Pretrain | VLA | 42,123,456 | VLA training stability |

**Report**: Mean ± std across seeds, min/max spread, learning curve envelope plot.

### T7.3 Mixed Precision Validation

| ID | Task | Precision | Purpose |
|----|------|-----------|---------|
| T7.3a | VLA Pretrain | fp32 | Baseline numerical accuracy |
| T7.3b | VLA Pretrain | bf16 | Speed vs accuracy tradeoff |
| T7.3c | PPO Go1 | fp32 vs bf16 | RL stability under mixed precision |

**Report**: Loss curves (fp32 vs bf16), final metric difference, throughput gain.

---

## How to Report Results

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

When filing GitHub issues for test results:
- `benchmark-result` — successful test report
- `convergence-issue` — algorithm did not converge as expected
- `performance-gap` — result significantly below reference
- `bug` — crash, error, or incorrect behavior

---

## Priority Matrix

| Priority | Tests | Difficulty | Time | GPU |
|----------|-------|-----------|------|-----|
| 🔴 P0 (Critical) | T1.1, T1.2, T1.3 | Easy | 10-30 min | 1× any |
| 🟠 P1 (High) | T2.1, T3.1, T3.2 | Medium | 30-60 min | 1× GPU |
| 🟡 P2 (Medium) | T4.1, T5.1, T5.2 | Medium | 2-8 hrs | 1-8× GPU |
| 🟢 P3 (Nice-to-have) | T4.2, T4.3, T5.3, T6.x | Hard | 4-24 hrs | 2-8× GPU |
| 🔵 P4 (Advanced) | T7.x | Medium | 1-4 hrs | 1-8× GPU |
