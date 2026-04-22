# RoboRenForce Benchmark Plan

**Date**: 2026-04-22
**Goal**: Produce reproducible benchmark results for paper submission

---

## 1. Framework Capability Matrix (Paper Table 1)

### Training Paradigms

| Paradigm | Algorithm | Runner | Script | Test | Status |
|----------|-----------|--------|--------|------|--------|
| Pretrain (SL) | VLAPretrainAlgorithm | VLAPretrainRunner | `train_single_gpu.py` | ✅ 281 tests | **Ready** |
| Pretrain (DDP) | VLAPretrainAlgorithm | DistributedVLAPretrainRunner | `train_ddp.py` | ✅ 8-GPU verified | **Ready** |
| SFT | SFTAlgorithm (KL reg.) | VLASFTRunner | `train_sft.py` | ✅ 10 tests | **Ready** |
| SFT (DDP) | SFTAlgorithm | DistributedVLASFTRunner | `train_sft_ddp.py` | ✅ 2-GPU verified | **Ready** |
| GRPO | GRPOAlgorithm | VLAGRPORunner | `train_robotwin_grpo.py` | ✅ 13 tests | **Ready** |
| PPO (GAE) | PPOAlgorithm | VLAPPORunner | `train_robotwin_grpo.py --algo ppo` | ✅ | **Ready** |
| SAC | SACAlgorithm | — | — | ✅ unit tests | Code only |
| IQL | IQLAlgorithm | — | — | ✅ unit tests | Code only |
| DAgger | DAggerAlgorithm | — | — | ✅ unit tests | Code only |

### Supported VLM Backbones (Paper Table 2)

| Model | Params | Action Head | Pretrain | SFT | RL | Weights |
|-------|--------|------------|----------|-----|-----|---------|
| **Qwen2-VL** | 2B | Regression/Diffusion | ✅ | ✅ | ✅ | ✅ Downloaded (4.2GB) |
| **Qwen3-VL** | 2B | Regression/Diffusion | ✅ | ✅ | ✅ | ⚠️ Metadata only |
| **OpenPI (pi0.5)** | 4B | Flow matching | ✅ | ✅ | ✅ | ❌ Need download |
| **GR00T N1.7** | 3B | DiT | ✅ | ✅ | ✅ | ❌ Need download |
| **MLP Baseline** | ~1M | Regression | ✅ | ✅ | ✅ | Built-in |

### Benchmark Environments (Paper Table 3)

| Benchmark | Platform | Tasks | Robot | Data | Sim | Status |
|-----------|----------|-------|-------|------|-----|--------|
| **Psi0 Humanoid** | Offline | 14 tasks | G1 Dex3 | ✅ 14 datasets, 152K+ frames | N/A | **Ready** |
| **RoboTwin** | SAPIEN3 | 60+ tasks | Piper/ALOHA | Demo data | ✅ Installed | **Ready** |
| **Isaac Lab Loco** | Isaac Sim | 8+ robots | A1/Go1/Go2/H1/G1/Anymal | Sim-generated | ❌ Not installed | **Needs Isaac Sim** |
| **LeRobot Hub** | Offline | 50+ datasets | Various | ✅ HF download | N/A | **Ready** (download) |

---

## 2. Environment Readiness Checklist

### ✅ Available Now (No Setup Needed)

| Component | Version | Path/Notes |
|-----------|---------|------------|
| Python | 3.10.12 | `.venv/` managed by uv |
| PyTorch | 2.6.0+cu124 | CUDA 12.4 |
| GPU | 8× H100 80GB | 640GB total VRAM |
| transformers | 5.5.4 | Qwen2-VL/Qwen3-VL support |
| einops | 0.8.2 | Vision model dependency |
| safetensors | 0.7.0 | Checkpoint format |
| pyarrow | 23.0.1 | LeRobot Parquet I/O |
| tensorboard | 2.20.0 | Training logs |
| wandb | 0.26.0 | Experiment tracking |
| SAPIEN | 3.0.1 | RoboTwin physics |
| Qwen2-VL-2B weights | 4.2GB | `~/.cache/huggingface/hub/` |
| Psi0 G1 data (14 tasks) | ~152K frames | `/vepfs/users/zza/hvla/data/psi-data-shared/` |

### ⚠️ Optional — Download Required

| Component | Install Command | Size | Needed For |
|-----------|----------------|------|------------|
| Qwen3-VL-2B weights | `huggingface-cli download Qwen/Qwen3-VL-2B-Instruct` | ~4GB | Qwen3-VL benchmark |
| OpenPI pi0.5 weights | `huggingface-cli download lerobot/pi05_base` | ~8GB | OpenPI benchmark |
| GR00T N1.7 weights | `huggingface-cli download nvidia/GR00T-N1.7-3B` | ~6GB | GR00T benchmark |
| LeRobot PushT dataset | `python scripts/data/download_lerobot_dataset.py --repo lerobot/pusht` | ~2GB | Cross-dataset eval |
| peft (LoRA) | `pip install peft>=0.8.0` | ~10MB | LoRA fine-tuning |
| Isaac Lab | See `ENVIRONMENT-SETUP.md` | ~5GB | Locomotion benchmark |

---

## 3. Benchmark Experiment Plan

### Experiment A: Pretrain Scaling (Table 4 in paper)

**Goal**: Compare models × action heads on Psi0 G1 PickApple

| Exp | Model | Action Head | GPUs | Batch | Epochs | Metric |
|-----|-------|------------|------|-------|--------|--------|
| A1 | MLP Baseline | Regression | 1 | 32 | 50 | val_loss, action_error |
| A2 | Qwen2-VL-2B (frozen) | Regression | 8 | 16×8=128 | 20 | val_loss, action_error |
| A3 | Qwen2-VL-2B (frozen) | Diffusion | 8 | 16×8=128 | 20 | val_loss, action_error |
| A4 | Qwen3-VL-2B (frozen) | Regression | 8 | 16×8=128 | 20 | val_loss, action_error |
| A5 | OpenPI pi0.5 (frozen) | Flow | 8 | 8×8=64 | 20 | val_loss, action_error |
| A6 | GR00T N1.7 (frozen) | DiT | 8 | 8×8=64 | 20 | val_loss, action_error |

**Commands**:
```bash
# A1: MLP Baseline
python scripts/vla/pretrain/train_single_gpu.py \
    --mock --epochs 50 --batch_size 32

# A2: Qwen2-VL + Regression (8 GPU)
torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
    --data_root /vepfs/users/zza/hvla/data/psi-data-shared/unitree_dex3_converted/G1_Dex3_PickApple \
    --head regression --epochs 20 --batch_size 16

# A3: Qwen2-VL + Diffusion (8 GPU)
torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
    --data_root /vepfs/users/zza/hvla/data/psi-data-shared/unitree_dex3_converted/G1_Dex3_PickApple \
    --head diffusion --epochs 20 --batch_size 16
```

### Experiment B: SFT Fine-tune (Table 5)

**Goal**: Pretrained model → task-specific SFT, compare KL coefficient impact

| Exp | Base | KL coef | Freeze | Metric |
|-----|------|---------|--------|--------|
| B1 | A2 checkpoint | 0.0 | backbone | val_loss |
| B2 | A2 checkpoint | 0.01 | backbone | val_loss |
| B3 | A2 checkpoint | 0.1 | backbone | val_loss |
| B4 | A2 checkpoint | 0.0 | none (full FT) | val_loss |

### Experiment C: RL Fine-tune on RoboTwin (Table 6)

**Goal**: SFT model → RL, compare GRPO vs PPO

| Exp | Base | Algorithm | Envs | Iterations | Metric |
|-----|------|-----------|------|------------|--------|
| C1 | B2 checkpoint | GRPO | 8 | 500 | success_rate, mean_return |
| C2 | B2 checkpoint | PPO | 8 | 500 | success_rate, mean_return |
| C3 | Random init | GRPO | 8 | 500 | success_rate (ablation) |
| C4 | Random init | PPO | 8 | 500 | success_rate (ablation) |

### Experiment D: Multi-task Generalization (Table 7)

**Goal**: Train on N tasks, eval on held-out tasks

| Train Tasks | Eval Tasks | Model | Paradigm |
|------------|------------|-------|----------|
| 10 Psi0 tasks | 4 held-out Psi0 tasks | Qwen2-VL | Pretrain |
| 10 Psi0 tasks | 4 held-out Psi0 tasks | Qwen2-VL | Pretrain→SFT |

### Experiment E: DDP Scaling Efficiency (Table 8)

**Goal**: Throughput vs number of GPUs

| GPUs | Batch/GPU | Effective Batch | Time/Epoch | Speedup |
|------|-----------|----------------|------------|---------|
| 1 | 16 | 16 | baseline | 1.0× |
| 2 | 16 | 32 | ? | ? |
| 4 | 16 | 64 | ? | ? |
| 8 | 16 | 128 | ? | ? |

---

## 4. Paper Table Templates

### Table 1: Framework Comparison (RoboRenForce vs Others)

| Feature | RLinf | RoboRenForce | OpenVLA | Octo | LeRobot |
|---------|-------|-------------|---------|------|---------|
| **Training Paradigms** | | | | | |
| Supervised Pretrain | ✅ | ✅ | ✅ | ✅ | ✅ |
| SFT + KL | ✅ | ✅ | ❌ | ❌ | ❌ |
| PPO (on-policy RL) | ✅ | ✅ | ❌ | ❌ | ❌ |
| GRPO | ✅ | ✅ | ❌ | ❌ | ❌ |
| SAC / IQL / DAgger | ✅ | ✅ | ❌ | ❌ | ❌ |
| DPO (preference) | ✅ | ⏳ | ❌ | ❌ | ❌ |
| **VLM Backbones** | | | | | |
| Qwen2-VL | ✅ | ✅ | ❌ | ❌ | ❌ |
| Qwen3-VL | ✅ | ✅ | ❌ | ❌ | ❌ |
| OpenPI / pi0 | ✅ | ✅ | ❌ | ❌ | ✅ |
| GR00T | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Action Decoders** | | | | | |
| Regression (MLP) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Diffusion (DiT) | ✅ | ✅ | ❌ | ❌ | ✅ |
| Flow matching | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Architecture** | | | | | |
| 3-System Design | ✅ | ✅ | ❌ | ❌ | ❌ |
| ForwardType dispatch | ✅ | ✅ | ❌ | ❌ | ❌ |
| Configclass system | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Distributed** | | | | | |
| DDP | ✅ | ✅ | ✅ | ✅ | ✅ |
| FSDP / DeepSpeed | ✅ | ⏳ | ✅ | ❌ | ❌ |
| Mixed precision | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Environments** | | | | | |
| RoboTwin (60+ tasks) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Isaac Lab | ✅ | ✅ | ❌ | ❌ | ❌ |
| Real robot support | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Classic RL** | | | | | |
| Locomotion (PPO/SAC) | ❌ | ✅ (36 algos) | ❌ | ❌ | ❌ |
| World model training | ❌ | ✅ | ❌ | ❌ | ❌ |
| Imitation learning | ❌ | ✅ (GAIL/AMP) | ❌ | ❌ | ❌ |

### Table 2: Pretrain Results

| Model | Backbone | Head | Params (trainable) | Val Loss ↓ | Action MSE ↓ | Train Time |
|-------|----------|------|--------------------|-----------|-------------|------------|
| MLP Baseline | — | Regression | 1M | — | — | — |
| Qwen2-VL | 2B (frozen) | Regression | ~600K | — | — | — |
| Qwen2-VL | 2B (frozen) | Diffusion | ~2M | — | — | — |
| Qwen3-VL | 2B (frozen) | Regression | ~600K | — | — | — |
| OpenPI | 4B (frozen) | Flow | ~2M | — | — | — |
| GR00T | 3B (frozen) | DiT | ~2M | — | — | — |

### Table 3: RL Fine-tune Results

| Method | Pretrain | SFT | RL Algo | Success Rate ↑ | Mean Return ↑ |
|--------|---------|-----|---------|---------------|---------------|
| Random | ❌ | ❌ | GRPO | — | — |
| Random | ❌ | ❌ | PPO | — | — |
| Pretrain only | ✅ | ❌ | — | — | — |
| Pretrain + SFT | ✅ | ✅ | — | — | — |
| Pretrain + GRPO | ✅ | ❌ | GRPO | — | — |
| Pretrain + SFT + GRPO | ✅ | ✅ | GRPO | — | — |
| Pretrain + SFT + PPO | ✅ | ✅ | PPO | — | — |

---

## 5. Execution Priority

### Phase I: Quick Wins (1-2 days)

1. **Fix peft**: `pip install peft>=0.8.0`
2. **Run Exp A2**: Qwen2-VL real pretrain on G1 PickApple (8 GPU)
3. **Run Exp E**: DDP scaling test (1/2/4/8 GPU)
4. **Fill Table 2** with real numbers

### Phase II: Full Pipeline (3-5 days)

5. **Download missing weights**: Qwen3-VL, OpenPI, GR00T
6. **Run Exp A3-A6**: All model × head combinations
7. **Run Exp B1-B4**: SFT ablation
8. **Run Exp C1-C4**: RL fine-tune on RoboTwin (needs SAPIEN3 rendering)

### Phase III: Paper-ready (1 week)

9. **Run Exp D**: Multi-task generalization
10. **Run Exp C with real env**: RoboTwin + SAPIEN3
11. **Generate plots**: learning curves, scaling charts
12. **Write reproducibility section**: commands, configs, seeds

---

## 6. Reproducibility

### Seeds
All experiments use seed=42 for train/val split and initialization.

### Hardware
- GPU: 8× NVIDIA H100 80GB HBM3
- Storage: /vepfs shared filesystem
- Python: 3.10.12, PyTorch 2.6.0+cu124

### Checkpoints
All checkpoints saved in LeRobot-compatible format:
```
checkpoints/
├── pretrain/         # Phase A results
├── sft/              # Phase B results
├── rl/               # Phase C results
└── scaling/          # Phase E results
```
