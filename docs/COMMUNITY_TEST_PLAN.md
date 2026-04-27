# RoboRenForce VLA Test & Training Plan

> **Scope**: VLA pretrain → SFT → RL fine-tuning, end-to-end on 4 manipulation simulators.
> Locomotion, classic-control RL, D4RL offline, MJLab, IsaacLab, GAIL, MBPO are **out of scope** for this plan.

This document is both a **checklist** and a **tutorial**. New contributors should read sections 0 → 1 → the relevant section among 2/3/4 end-to-end before opening pull requests.

---

## Table of Contents

0. [Prerequisites](#0-prerequisites) — Python, system deps, Vulkan, env vars
1. [Quick Smoke (≤10 min)](#1-quick-smoke-10-min) — three-step verification
2. [VLA Pretrain](#2-vla-pretrain) — `scripts/vla/pretrain/`, single-GPU + DDP
3. [VLA SFT (Post-Train)](#3-vla-sft-post-train) — `scripts/vla/post_train/`
4. [VLA RL Fine-Tuning](#4-vla-rl-fine-tuning) — LIBERO / ManiSkill / CALVIN / RoboTwin
5. [VLM Backbones](#5-vlm-backbones) — Qwen2-VL, Qwen3-VL, OpenPI, GR00T
6. [Datasets — Formats & Download](#6-datasets--formats--download) — LeRobot v2, Psi0, per-sim demo data
7. [How to Report Results](#7-how-to-report-results)
8. [Troubleshooting](#8-troubleshooting)

---

## 0. Prerequisites

### 0.1 System Packages

```bash
apt-get install -y \
  python3.10-dev ffmpeg git-lfs \
  libvulkan1 vulkan-tools mesa-vulkan-drivers \
  libegl1 libgles2-mesa
git lfs install
```

| Package | Why |
|---|---|
| `python3.10-dev` | Cython builds (mplib, sapien) |
| `ffmpeg` | LeRobot video I/O |
| `git-lfs` | HF dataset weights, RoboTwin/LIBERO assets |
| `libvulkan1` + `vulkan-tools` | SAPIEN GPU rendering (ManiSkill, RoboTwin) |
| `mesa-vulkan-drivers` | CPU Vulkan fallback (llvmpipe) |
| `libegl1`, `libgles2-mesa` | Headless EGL fallback |

**Vulkan ICD discovery (NVIDIA hosts).** SAPIEN looks for the NVIDIA ICD at `/usr/share/vulkan/icd.d/nvidia_icd.json`. The driver package places it at `/etc/vulkan/icd.d/nvidia_icd.json`. Symlink:

```bash
ln -sf /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
vulkaninfo --summary | grep -A1 deviceName    # should list NVIDIA GPU(s)
```

If the loader cannot create a Vulkan instance against the NVIDIA driver, ManiSkill `obs_mode=rgbd` and RoboTwin will hang forever in scene init. Fall back to ManiSkill `--obs_mode state` until Vulkan is healthy (RoboTwin has no equivalent fallback).

### 0.2 Python Environment

```bash
cd RoboRenForce
uv venv --python 3.10 .venv
. .venv/bin/activate
uv pip install -e source/RoboRenForce
```

### 0.3 Per-Stage Install Matrix

Install only what you need for the stage you are running:

| Stage | Packages |
|---|---|
| §2 Pretrain (mock) | core only |
| §2 Pretrain (Qwen2-VL) | `uv pip install transformers accelerate` |
| §3 SFT | core only (uses LeRobot dataset) |
| §4 LIBERO | `uv pip install -e source/tasks/RRF_libero source/tasks/RRF_libero_vla_rl` + `pip install robosuite` + clone & install LIBERO |
| §4 ManiSkill | `uv pip install -e source/tasks/RRF_maniskill source/tasks/RRF_maniskill_vla_rl` + `uv pip install mani_skill` |
| §4 CALVIN | `uv pip install -e source/tasks/RRF_calvin source/tasks/RRF_calvin_vla_rl` + clone & install `calvin_env` |
| §4 RoboTwin | `uv pip install -e source/tasks/RRF_robotwin source/tasks/RRF_robotwin_vla_rl` + clone RoboTwin assets, `uv pip install mplib==0.2.1 toppra` |
| §4 distributed | `uv pip install -e source/RRF_orchestra` |

### 0.4 Environment Variables

| Variable | When | Value |
|---|---|---|
| `HF_TOKEN` | Pretrain on Psi0 / NVIDIA gated repos | `hf_xxx_…` (see [SETUP_GUIDE.md §3](SETUP_GUIDE.md#3-huggingface-gated-repos)) |
| `HF_HOME` | Always recommended | path on a roomy filesystem (>200 GB) |
| `ASSETS_PATH` | RoboTwin (§4) | RoboTwin repo **root** path (not `…/assets/`) |
| `SAPIEN_HEADLESS` | RoboTwin/ManiSkill on no-display hosts | `1` |

### 0.5 Disk Layout

`/tmp` on shared dev hosts is typically <20 GB. **Always** keep model checkpoints, HF caches, and dataset extracts on a roomy filesystem (e.g. `/vepfs`):

```bash
export HF_HOME=/vepfs/$USER/.hf_cache
mkdir -p /vepfs/$USER/checkpoints /vepfs/$USER/data
```

---

## 1. Quick Smoke (≤10 min)

Three commands that exercise each pipeline with mock or trivial datasets. All must succeed before opening a PR that touches VLA code.

### 1.1 Pretrain mock

```bash
python scripts/vla/pretrain/train_single_gpu.py --mock --epochs 1 --batch_size 4
```

Expected: 1 epoch, action loss < 1.0, checkpoint written to `checkpoints/vla_pretrain/checkpoint_final.pt`.

### 1.2 SFT mock

```bash
python scripts/vla/post_train/train_sft.py --mock --epochs 1 --batch_size 4
```

Expected: action loss decreasing across log lines, checkpoint at `checkpoints/sft/checkpoint_final.pt`.

### 1.3 RL smoke (pick whichever simulator is available)

```bash
# State-only (no Vulkan needed) — fastest
python scripts/vla/rl/train_maniskill.py --task ManiSkill-PickCube-GRPO-v0 \
       --num_envs 2 --max_iterations 1 --obs_mode state

# Robosuite/MuJoCo (no Vulkan needed)
python scripts/vla/rl/train_libero.py --task LIBERO-Spatial-GRPO-v0 \
       --num_envs 2 --max_iterations 1
```

Expected: 1 GRPO iteration, no exceptions, log dir under `logs/RFRL/<task>/<timestamp>/`.

---

## 2. VLA Pretrain

Pretrain a VLA actor (VLM backbone + action head) on offline LeRobot-format demonstrations.

**Scripts:** `scripts/vla/pretrain/train_single_gpu.py`, `scripts/vla/pretrain/train_ddp.py`
**Runner:** `VLAPretrainRunner` (`source/RoboRenForce/RoboRenForce/runners/vla/pretrain/vla_pretrain_runner.py`)
**Algorithm:** `VLAPretrainAlgorithm` (`source/RoboRenForce/RoboRenForce/algorithms/vla_training/pretrain_algorithm.py`) — MSE or diffusion loss, optional AMP.

### 2.1 Single-GPU

```bash
python scripts/vla/pretrain/train_single_gpu.py \
    --data_root /vepfs/$USER/data/psi-data-shared/unitree_dex3_converted/G1_Dex3_PickApple \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --head regression \
    --action_dim 36 \
    --image_size 224 224 \
    --batch_size 8 --epochs 10 --lr 1e-4 --amp \
    --checkpoint_dir /vepfs/$USER/checkpoints/vla_pretrain/G1_PickApple_Qwen2VL2B
```

**Modes** (mutually exclusive):
- `--mock` — MockVLM + MockDataset; CPU-friendly, no model download.
- `--psi0` — MockVLM + real Psi0 LeRobot data; exercises the data pipeline without loading a 2B-parameter VLM.
- *(default)* — Qwen2-VL backbone + LeRobot data.

**Key flags:**

| Flag | Default | Notes |
|---|---|---|
| `--data_root` | `data/example_dataset` | LeRobot v2 directory (must contain `meta/info.json`, `data/chunk-…/file-….parquet`) |
| `--frames_dir` | `""` | Pre-extracted frames (skip mp4 decoding) |
| `--head` | `regression` | `regression` (MLP) or `diffusion` |
| `--action_dim` | `36` | Match dataset (G1 Dex3 = 36; 7-DoF Franka = 7) |
| `--amp` | off | bfloat16 AMP; required for Qwen2-VL-7B on a single 80 GB GPU |
| `--save_interval` | `500` | Steps between checkpoints |
| `--resume` | `None` | Resume from `checkpoint_step_<N>.pt` |

**Output checkpoint structure** (loaded by SFT/RL via `--checkpoint`):
```
{
  "vla_actor_state_dict": {...},
  "optimizer_state_dict":  {...},
  "scheduler_state_dict":  {...},
  "global_step": int,
}
```

### 2.2 Multi-GPU DDP

```bash
torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
    --data_root /vepfs/$USER/data/psi-data-shared/unitree_dex3_converted/G1_Dex3_PickApple \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --batch_size 8 --epochs 50 --lr 1e-4 --amp \
    --checkpoint_dir /vepfs/$USER/checkpoints/vla_pretrain/G1_PickApple_Qwen2VL2B_ddp
```

`train_ddp.py` wraps the same algorithm in `DistributedDataParallel`, with a `DistributedSampler` ensuring each rank sees a non-overlapping shard.

### 2.3 Recommended pretrain matrix

| Backbone | Params | GPUs | Batch / GPU | Notes |
|---|---|---|---|---|
| Mock | 64 ch | 1× any | 4 | Pipeline smoke (§1.1) |
| Qwen2-VL-2B | 2 B | 1× H100 80 GB | 8 | Default; AMP recommended |
| Qwen2-VL-7B | 7 B | 4× H100 80 GB DDP | 4 | AMP **required**, frozen backbone |
| Qwen3-VL-2B | 2 B | 1× H100 80 GB | 8 | See §5 |
| OpenPI π0.5 | 3 B | 2× H100 80 GB DDP | 4 | Flow-matching head; see §5 |
| GR00T N1.7 | 8 B | 4× H100 80 GB DDP | 2 | Cosmos-Reason2 backbone (gated, see [SETUP_GUIDE.md §3](SETUP_GUIDE.md#3-huggingface-gated-repos)) |

---

## 3. VLA SFT (Post-Train)

Fine-tune a pretrained VLA actor on a task-specific demonstration set, optionally with KL regularization against the pretrained reference policy.

**Scripts:** `scripts/vla/post_train/train_sft.py` (single-GPU), `scripts/vla/post_train/train_sft_ddp.py` (DDP)
**Runner:** `VLASFTRunner` (`source/RoboRenForce/RoboRenForce/runners/vla/post_train/sft_runner.py`)
**Algorithm:** `SFTAlgorithm` (`source/RoboRenForce/RoboRenForce/algorithms/vla_training/sft.py`) — MSE / L1 / smooth-L1 action loss + optional KL term.

### 3.1 Single-GPU

```bash
python scripts/vla/post_train/train_sft.py \
    --pretrained /vepfs/$USER/checkpoints/vla_pretrain/G1_PickApple_Qwen2VL2B/checkpoint_final.pt \
    --data_root /vepfs/$USER/data/libero_spatial_demos \
    --action_loss_type mse \
    --kl_coef 0.01 \
    --freeze_backbone \
    --epochs 30 --batch_size 16 --lr 1e-4 \
    --checkpoint_dir /vepfs/$USER/checkpoints/sft/libero_spatial
```

**Key flags:**

| Flag | Default | Notes |
|---|---|---|
| `--pretrained` | `""` | Pretrain checkpoint (`*.pt`) |
| `--data_root` | `""` | LeRobot v2 task data (empty → MockSFT) |
| `--action_loss_type` | `mse` | `mse` / `l1` / `smooth_l1` |
| `--kl_coef` | `0.0` | Set >0 to enable KL-vs-reference regularization |
| `--freeze_backbone` | off | Freeze VLM, train action head only (cheap, 10× faster) |
| `--scheduler` | `warmup` | `warmup` (linear) or `cosine` |
| `--model_type` | `mlp_baseline` | Resolved through `RRF_models.get_model` |

### 3.2 KL regularization

When `--kl_coef > 0`, the runner loads the pretrained checkpoint as a frozen reference policy and adds `kl_coef * KL(π‖π_ref)` to the action loss. This stabilizes SFT when the new dataset distribution differs significantly from pretrain.

### 3.3 SFT output

Same checkpoint format as pretrain (`vla_actor_state_dict`); consumed directly by §4 RL via `--checkpoint`.

---

## 4. VLA RL Fine-Tuning

Online RL over 4 manipulation simulators. The CLI is uniform via `scripts/vla/rl/_common.py`:

```
python scripts/vla/rl/train_<sim>.py \
    --task <TASK_ID> \
    --num_envs <N> \
    --max_iterations <K> \
    --device cuda:0 \
    [--checkpoint <pretrain_or_sft_ckpt>] \
    [--obs_mode state]                       # ManiSkill only
```

`_common.run` looks up the task via `gymnasium.spec(task_id)`, builds the env from `env_cfg_entry_point.build()`, builds the policy from `runner_cfg.build_policy()`, and calls `runner.learn(num_iterations=K)`.

### 4.1 LIBERO

**Stack:** robosuite / MuJoCo (no Vulkan).
**Task IDs:** `LIBERO-Spatial-GRPO-v0`
**Env cfg:** `LiberoSpatialEnvCfg` (`source/tasks/RRF_libero_vla_rl/RRF_libero_vla_rl_tasks/spatial_pick_object/env_cfg.py`)
**Action dim:** 7 (Franka EEF Δpose + gripper) · **State dim:** 7 · **Max steps:** 300
**Smoke status:** ✅ pass (~335 s for 1 GRPO iter, num_envs=8)

**Install:**
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO
uv pip install -e LIBERO robosuite
uv pip install -e source/tasks/RRF_libero source/tasks/RRF_libero_vla_rl
```

**Datasets (RL is online — these are only for prior pretrain/SFT, optional):**
- `libero_spatial` (90 spatial-reasoning tasks) — bundled in `LIBERO/libero/datasets/`, materialise via `git lfs checkout` inside the LIBERO repo.
- Pre-converted LeRobot v2 mirror: `huggingface.co/datasets/lerobot/libero_*` — `huggingface-cli download lerobot/libero_spatial --local-dir /vepfs/$USER/data/libero_spatial`.

**Run:**
```bash
python scripts/vla/rl/train_libero.py \
    --task LIBERO-Spatial-GRPO-v0 --num_envs 8 --max_iterations 100 \
    --checkpoint /vepfs/$USER/checkpoints/sft/libero_spatial/checkpoint_final.pt
```

### 4.2 ManiSkill

**Stack:** SAPIEN 3 (Vulkan + CUDA-Vulkan interop, GPU-accelerated parallel sim).
**Task IDs:** `ManiSkill-PickCube-GRPO-v0`
**Env cfg:** `ManiSkillPickCubeEnvCfg` (`source/tasks/RRF_maniskill_vla_rl/RRF_maniskill_vla_rl_tasks/pick_cube/env_cfg.py`)
**Action dim:** 7 · **State dim:** 25 (state mode); rgbd adds `[B, H, W, 3]` images · **Max steps:** 200
**Smoke status:** ✅ state-only (~93 s); ⚠ rgbd requires healthy Vulkan ICD (see §0.1)

**Install:**
```bash
uv pip install mani_skill
uv pip install -e source/tasks/RRF_maniskill source/tasks/RRF_maniskill_vla_rl
```

**Datasets:** ManiSkill scenes are procedurally generated — **no external download is required for RL**. For optional pretrain demonstrations:
- ManiSkill 2 demonstrations: `python -m mani_skill.utils.download_demo PickCube-v1 -o /vepfs/$USER/data/maniskill_demos`.

**Run:**
```bash
# Full GPU rendering (Vulkan healthy)
python scripts/vla/rl/train_maniskill.py \
    --task ManiSkill-PickCube-GRPO-v0 --num_envs 16 --max_iterations 200

# State-only fallback (no Vulkan needed)
python scripts/vla/rl/train_maniskill.py \
    --task ManiSkill-PickCube-GRPO-v0 --num_envs 16 --max_iterations 200 \
    --obs_mode state
```

`obs_mode`/`control_mode`/`reward_mode` are dataclass fields on `ManiSkillPickCubeEnvCfg`; only `obs_mode` is plumbed through `_common.py` because it's the rendering escape hatch.

### 4.3 CALVIN

**Stack:** PyBullet + Hydra/OmegaConf (no Vulkan).
**Task IDs:** `CALVIN-D-GRPO-v0`
**Env cfg:** `CalvinDSplitEnvCfg` (`source/tasks/RRF_calvin_vla_rl/RRF_calvin_vla_rl_tasks/d_split/env_cfg.py`)
**Action dim:** 7 (6D EEF + 1D discrete gripper, thresholded at 0) · **State dim:** 7 · **Max steps:** 360
**Smoke status:** ✅ pass (~339 s)

**Install:**
```bash
git clone --recurse-submodules https://github.com/mees/calvin
uv pip install -e calvin/calvin_env
uv pip install -e source/tasks/RRF_calvin source/tasks/RRF_calvin_vla_rl
```

**Datasets:**
- The CALVIN env can be **constructed without a dataset** via the bundled `config_data_collection` Hydra config (already wired in `RRF_calvin/.../calvin_env.py`). Useful for CI / smoke.
- For real RL training and the published 5-subtask language-conditioned eval, download CALVIN scene D:
  ```bash
  bash calvin/dataset/download_data.sh D
  # → calvin/dataset/task_D_D/{training,validation}/
  ```
- Pass `--dataset_path` (mapped to `env_cfg.dataset_path`) when invoking RL.

**Run:**
```bash
# Smoke (no dataset required)
python scripts/vla/rl/train_calvin.py \
    --task CALVIN-D-GRPO-v0 --num_envs 4 --max_iterations 1

# Real (with downloaded dataset)
python scripts/vla/rl/train_calvin.py \
    --task CALVIN-D-GRPO-v0 --num_envs 4 --max_iterations 100 \
    --checkpoint /vepfs/$USER/checkpoints/sft/calvin/checkpoint_final.pt
```

### 4.4 RoboTwin

**Stack:** SAPIEN 3 + mplib + toppra (Vulkan **required** for scene init — no fallback).
**Task ID:** `RoboTwin-PlaceCup-GRPO-v0`
**Env cfg:** `RoboTwinPlaceCupEnvCfg` (`source/tasks/RRF_robotwin_vla_rl/RRF_robotwin_vla_rl_tasks/place_empty_cup/env_cfg.py`)
**Action dim:** 14 (bimanual) · **State dim:** 14 · **Max steps:** 200 · **Embodiment:** `["piper", "piper", 0.6]`
**Smoke status:** ⚠ blocked on hosts where Vulkan rendering hangs (see §8 troubleshooting)

**Install:**
```bash
git clone https://github.com/TianxingChen/RoboTwin /vepfs/$USER/code/RoboTwin
uv pip install mplib==0.2.1 toppra
uv pip install -e source/tasks/RRF_robotwin source/tasks/RRF_robotwin_vla_rl
export ASSETS_PATH=/vepfs/$USER/code/RoboTwin       # repo ROOT, not …/assets/
```

**Datasets:**
- Asset bundle (URDFs, meshes, scenes): cloned with the repo above + `git lfs checkout`.
- Pretrain/SFT demonstrations: convert RoboTwin's scripted-policy episodes to LeRobot v2 via `scripts/data/robotwin_to_lerobot.py` (see [DATA_DOWNLOAD.md](DATA_DOWNLOAD.md)).

**Run:**
```bash
python scripts/vla/rl/train_robotwin.py \
    --task RoboTwin-PlaceCup-GRPO-v0 --num_envs 4 --max_iterations 50 \
    --checkpoint /vepfs/$USER/checkpoints/sft/robotwin/checkpoint_final.pt
```

### 4.5 Distributed RL (RRF_orchestra)

For production RL fine-tuning of 7B+ VLA actors, use the rollout / inference / learner split runner:

```bash
uv pip install -e source/RRF_orchestra
python scripts/vla/rl/orchestra_<sim>.py --task <TASK_ID> --rollout_workers 4 --learner_gpus 4
```

See `source/RRF_orchestra/README.md` for the full topology.

---

## 5. VLM Backbones

Models live under `source/RoboRenForce/RoboRenForce/networks/vlm/`. All wrap upstream HuggingFace weights via a `@configclass`-style cfg.

| Backbone | File | Default checkpoint | Hidden dim | Gated? | Tested |
|---|---|---|---|---|---|
| Qwen2-VL-2B / 7B | `qwen2vl.py` | `Qwen/Qwen2-VL-2B-Instruct` | 1536 / 3584 | No | ✅ pretrain + SFT + RL |
| Qwen3-VL-2B | `qwen3vl.py` | `Qwen/Qwen3-VL-2B-Instruct` | 2048 | No | ✅ pretrain (smoke) |
| OpenPI π0.5 | `openpi.py` | community releases | varies | No | ✅ pretrain (smoke) |
| GR00T N1.7 | `gr00t.py` | `nvidia/GR00T-N1.7-8B` | varies | **Yes — request access** | ✅ pretrain on Cosmos-Reason2 |
| Mock | `mock.py` | n/a | configurable | No | smoke only |

The fusion layer (`fusion_layers.py`) concatenates VLM features with proprioception and projects to the action-head input dim.

For HF gated repos (GR00T, Cosmos-Reason2): see [SETUP_GUIDE.md §3](SETUP_GUIDE.md#3-huggingface-gated-repos).

---

## 6. Datasets — Formats & Download

### 6.1 LeRobot v2 (canonical for pretrain & SFT)

```
<data_root>/
├── meta/
│   ├── info.json          # episode count, fps, image specs
│   ├── episodes.jsonl     # per-episode metadata (length, task)
│   ├── tasks.jsonl        # task-id → language instruction
│   └── modality.json      # (optional, GR00T-only) state/action dim mapping
├── data/
│   └── chunk-000/
│       └── file-000.parquet   # (states, action, episode_index, frame_index, …)
└── videos/
    └── chunk-000/
        └── observation.images.<cam>/
            └── episode_000000.mp4
```

The reader is `LeRobotDataset` (`source/RoboRenForce/RoboRenForce/dataset/lerobot/lerobot_dataset.py`). It supports:
- Lazy video decoding or pre-extracted frames (`frames_dir=…`).
- Column remapping: `"states"` ↔ `"observation.state"`, `"proprioception"`.
- Mixture sampling across multiple LeRobot roots via `MixtureDataset` / `BatchMixtureSampler` (`mixture.py`).

### 6.2 Dataset sources

#### Psi0 (G1 Dex3 hand, 14 tasks, ~152 K frames)

```bash
# HuggingFace (gated — request access on the dataset page)
hf auth login --token $HF_TOKEN
python scripts/data/download_psi0_dataset.py \
    --split simple \
    --target /vepfs/$USER/data/psi-data-shared
# Output: /vepfs/$USER/data/psi-data-shared/unitree_dex3_converted/G1_Dex3_<task>/
```

Splits:
- `real` — bag-recorded teleop episodes
- `simple` — synthetic / scripted episodes (recommended for pretrain)
- `simple-eval` — held-out eval split

#### LIBERO

```bash
# In-tree (libero_spatial, libero_object, libero_goal, libero_10, libero_90)
cd LIBERO && git lfs checkout
ls libero/datasets/

# OR pre-converted LeRobot v2:
huggingface-cli download lerobot/libero_spatial \
    --repo-type dataset --local-dir /vepfs/$USER/data/libero_spatial
```

#### ManiSkill demonstrations (optional, for pretrain only)

```bash
python -m mani_skill.utils.download_demo PickCube-v1 \
    -o /vepfs/$USER/data/maniskill_demos
```

#### CALVIN scenes (required for the published 5-subtask eval)

```bash
cd calvin
bash dataset/download_data.sh D       # ~165 GB; or A/B/C/ABCD
ls dataset/task_D_D/training/         # 6 hours of teleop @ 30 Hz
```

#### RoboTwin scripted-policy demos (optional)

```bash
cd /vepfs/$USER/code/RoboTwin
git lfs checkout                        # asset MP4s + URDFs
# Generate fresh demos with the upstream scripted policy:
python scripts/collect_demos.py --task place_empty_cup --episodes 100
# Convert to LeRobot v2:
python scripts/data/robotwin_to_lerobot.py \
    --in /vepfs/$USER/code/RoboTwin/data/place_empty_cup \
    --out /vepfs/$USER/data/robotwin_place_empty_cup
```

### 6.3 Recommended placement

```
/vepfs/$USER/data/
├── psi-data-shared/                    # pretrain
│   └── unitree_dex3_converted/
├── libero_spatial/                     # SFT for §4.1
├── maniskill_demos/                    # optional pretrain
├── calvin_task_D/                      # §4.3 real-data eval
└── robotwin_place_empty_cup/           # SFT for §4.4
```

See [DATA_DOWNLOAD.md](DATA_DOWNLOAD.md) for the GR00T-flavored variant (`meta/modality.json`).

---

## 7. How to Report Results

Per-PR report template (paste into the PR body):

```markdown
### Stage tested
- [ ] §2.1 Pretrain single-GPU
- [ ] §2.2 Pretrain DDP
- [ ] §3 SFT
- [ ] §4.1 LIBERO RL
- [ ] §4.2 ManiSkill RL
- [ ] §4.3 CALVIN RL
- [ ] §4.4 RoboTwin RL

### Hardware
- GPUs: 1× H100 80 GB (or whatever)
- Driver / CUDA: 535.129 / 12.2

### Command
`python scripts/vla/rl/train_libero.py --task LIBERO-Spatial-GRPO-v0 --num_envs 8 --max_iterations 100`

### Result
- Wall-clock: 47 min for 100 iters
- Final action loss / reward: …
- Log dir: `logs/RFRL/LIBERO-Spatial-GRPO-v0/2026-04-27_…`
```

Attach the tensorboard event file or a screenshot of the loss curve.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: libvulkan.so.1: cannot open shared object` | Vulkan loader not installed | `apt-get install libvulkan1` (§0.1) |
| `Failed to find Vulkan ICD file` warning, then renderer hangs | NVIDIA ICD at `/etc/vulkan/icd.d/` not visible to SAPIEN | Symlink to `/usr/share/vulkan/icd.d/` (§0.1) |
| ManiSkill `obs_mode=rgbd` hangs >5 min at first step | Vulkan loader OK but GPU render pipeline stuck (e.g. headless EGL waiting on display) | Fall back to `--obs_mode state`; on remote hosts check `XDG_RUNTIME_DIR`, `DISPLAY` |
| RoboTwin: `setup_task` runs forever, `trial_seed` keeps incrementing | Scene-init exception swallowed by retry loop in `RoboTwin/robotwin/envs/vector_env.py` | Run with `py-spy dump --pid <P>` to surface the silenced exception; usually missing `ASSETS_PATH` or Vulkan |
| `TypeError: ASSETS_PATH=None` in RoboTwin init | `ASSETS_PATH` env var unset | `export ASSETS_PATH=/vepfs/$USER/code/RoboTwin` (the **root**, not `…/assets/`) |
| `ModuleNotFoundError: mplib.sapien_utils` | mplib 0.1.x missing the SAPIEN bridge | `uv pip install mplib==0.2.1` |
| `KeyError: 'task'` from CALVIN | Dataset path missing `.hydra/config.yaml` | Pass valid `--dataset_path` or rely on bundled `config_data_collection` (auto-fallback in `RRF_calvin`) |
| Triton `cannot import _C` during pretrain | gcc stderr suppressed by Triton | See [feedback memory](../README.md#triton-debug) — monkey-patch `triton.runtime.build._build` |
| HF gated repo 401 | Token expired or wrong | `hf auth login --token $HF_TOKEN`; check repo access at `huggingface.co/<repo>/settings` |
| OOM during Qwen2-VL-7B pretrain | bf16 not enabled | Add `--amp`; consider `--freeze_backbone` for SFT |
| `device mismatch cuda:0 vs cpu` on done mask (ManiSkill) | Already fixed in `RRF_maniskill/.../maniskill_env.py`; if you see this, you're on an older revision | `git pull` |

### 8.1 Diagnosing a SAPIEN hang with py-spy

```bash
pip install py-spy
py-spy dump --pid <stuck_pid> --native --locals | head -80
```

Look for frames in `libnvidia-eglcore.so` doing `poll` — that's a GPU sync wait, almost always Vulkan-loader / ICD related.

---

*Last updated: 2026-04-27. Smoke status:* LIBERO ✅, ManiSkill ✅ (state), CALVIN ✅, RoboTwin ⚠ (Vulkan rendering issue on the current host — see §0.1, §8).
