# GR00T Setup & Finetune Guide

A community-facing recipe for downloading **NVIDIA Isaac GR00T N1.7-3B**,
preparing the environment, and running an end-to-end finetune on the bundled
SO101 demo dataset. Every gotcha listed below has been hit and verified — the
errors are usually opaque tracebacks deep inside `torch` / `triton` /
`torchcodec`, so the troubleshooting table at the bottom is worth skimming
before you start.

---

## 1. System packages

A minimal Linux container is missing four packages that block the GR00T
finetune pipeline. Install them all up front:

```bash
sudo apt-get update && sudo apt-get install -y \
    python3.10-dev      \
    ffmpeg              \
    libaio-dev          \
    git-lfs
```

| Package          | What it unblocks                                                                                                                         |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `python3.10-dev` | Triton JIT-compiles `cuda_utils.so` against `Python.h` on the **first GPU touch**. Without the headers: `fatal error: Python.h: No such file`. |
| `ffmpeg`         | Provides `libavutil.so.56`. `torchcodec` (the only video backend supported by `isaac-gr00t`) fails to load: `Video backend 'torchcodec' is not available`. |
| `libaio-dev`     | DeepSpeed CPU-offload and several flash-attn helper builds need it.                                                                      |
| `git-lfs`        | The `demo_data/*.parquet` and `*.mp4` files ship as ~512-byte LFS pointers. Without LFS the dataloader returns garbage tensors.          |

After installing `git-lfs`, materialise the demo data:

```bash
cd third_party/isaac-gr00t
git lfs install
git lfs checkout            # ~617 MB — turns the 512-byte pointers into real files
```

---

## 2. Python environment

`isaac-gr00t` ships its own `uv`-managed venv (`pyproject.toml` pinned). Use it
— the finetune script depends on `tyro`, `flash-attn`, and a specific
`torchcodec` build that the system Python won't have.

```bash
cd third_party/isaac-gr00t
uv sync                     # creates .venv/ with pinned deps
source .venv/bin/activate
python -c "import gr00t; print(gr00t.__file__)"   # smoke check
```

---

## 3. HuggingFace authentication (gated weights)

GR00T-N1.7 pulls **`nvidia/Cosmos-Reason2-2B`** as the VLM backbone — that
repo is gated. To unblock:

1. Visit <https://huggingface.co/nvidia/Cosmos-Reason2-2B> and click
   **"Request access"**. Approval is automatic for most accounts.
2. Mint a read token at <https://huggingface.co/settings/tokens>.
3. Authenticate, **and** export the token as an env var (subprocesses
   spawned by the trainer don't always pick up the cached token file):

   ```bash
   hf auth login --token hf_xxx_your_token_here
   export HF_TOKEN=hf_xxx_your_token_here
   ```

4. (Optional) Point the HF cache at a fast filesystem with plenty of room —
   GR00T-N1.7-3B is ~6 GB and Cosmos-Reason2-2B is ~4 GB:

   ```bash
   export HF_HOME=/path/to/large_disk/.hf_cache
   ```

---

## 4. Downloading the GR00T weights

Two equivalent ways. **Pre-downloading** is recommended because the first
training step would otherwise spend several minutes pulling weights and is
hard to resume cleanly if the connection drops.

### Option A — `huggingface_hub` Python API

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="nvidia/GR00T-N1.7-3B")
snapshot_download(repo_id="nvidia/Cosmos-Reason2-2B")   # gated; needs HF_TOKEN
```

### Option B — `hf` CLI

```bash
hf download nvidia/GR00T-N1.7-3B
hf download nvidia/Cosmos-Reason2-2B
```

After download, the snapshot lives at:

```
$HF_HOME/hub/models--nvidia--GR00T-N1.7-3B/snapshots/<commit_hash>/
```

Capture the path — you'll pass it to `--base-model-path`:

```bash
GROOT_PATH=$(hf cache scan --quiet | grep GR00T-N1.7-3B | awk '{print $NF}')
```

---

## 5. Disk space

A single GR00T checkpoint is **~12.6 GB** (sharded safetensors) plus
**~1 GB** of optimizer state. The dataloader also caches video shards under
`$TMPDIR`. Plan for:

| Item                            | Approx size        |
| ------------------------------- | ------------------ |
| GR00T-N1.7-3B weights           | 6 GB               |
| Cosmos-Reason2-2B weights       | 4 GB               |
| One training checkpoint         | 13–14 GB           |
| Video shard cache per epoch     | a few GB           |

If your `/tmp` is a small tmpfs / overlay (often the case in containers),
point both training output and `TMPDIR` at a roomy filesystem:

```bash
export TMPDIR=/path/to/large_disk/scratch
mkdir -p "$TMPDIR"
```

A failed save halfway through a 13 GB checkpoint write is the most common
"I had it working then it crashed" symptom — almost always a `/tmp` capacity
issue.

---

> **Need a larger or different dataset?** See **[DATA_DOWNLOAD.md](DATA_DOWNLOAD.md)** for
> bundled demo data, downloading NVIDIA's GR00T-flavored datasets from HuggingFace
> (Teleop-G1, Teleop-GR1, X-Embodiment-Sim, etc.), and converting community
> LeRobot v3 datasets to GR00T-flavored v2.

---

## 6. Verified finetune recipe (smoke test)

Trains for 20 steps on the bundled `cube_to_bowl_5` SO101 demo. Loss should
go from ~1.10 → ~1.04 in roughly 90 seconds on a single H100/H800-class GPU.
Final checkpoint lands in `--output-dir`.

```bash
cd third_party/isaac-gr00t
source .venv/bin/activate

export HF_TOKEN=hf_xxx_your_token_here
export HF_HOME=/path/to/large_disk/.hf_cache
export TMPDIR=/path/to/large_disk/scratch
export CUDA_VISIBLE_DEVICES=0
export USE_WANDB=0
export MAX_STEPS=20
export SAVE_STEPS=20
export GLOBAL_BATCH_SIZE=2
export DATALOADER_NUM_WORKERS=0

GROOT_PATH=$(hf cache scan --quiet | grep GR00T-N1.7-3B | awk '{print $NF}')

bash examples/finetune.sh \
    --base-model-path       "$GROOT_PATH" \
    --dataset-path          "$PWD/demo_data/cube_to_bowl_5" \
    --modality-config-path  "$PWD/examples/SO100/so100_config.py" \
    --embodiment-tag        new_embodiment \
    --output-dir            /path/to/large_disk/gr00t_smoke_out
```

### Two non-obvious flags

- **`--modality-config-path` expects a Python file**, not the dataset's
  `meta/modality.json`. The Python file calls `register_modality_config(...)`
  to wire dataset keys (`observation.images.front`, `action.gripper`, etc.)
  into the model's expected modality slots. For SO101-style demos
  (single arm + gripper, front + wrist cameras), `examples/SO100/so100_config.py`
  matches the keys in `cube_to_bowl_5` directly.
- **`--embodiment-tag` must match the tag the modality config registers
  under** (`EmbodimentTag.NEW_EMBODIMENT` → `--embodiment-tag new_embodiment`).
  A mismatch yields a confusing "no projector head found" error.

For a real run, scale `MAX_STEPS`, raise `GLOBAL_BATCH_SIZE` to fill VRAM,
and set `DATALOADER_NUM_WORKERS=4+`.

---

## 7. Troubleshooting

| Symptom                                                          | Fix                                                                                       |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `fatal error: Python.h: No such file` (during triton compile)    | `apt-get install -y python3.10-dev`                                                       |
| `Video backend 'torchcodec' is not available`                    | `apt-get install -y ffmpeg` (provides `libavutil.so.56`)                                  |
| `Repository not found` / `401` on `nvidia/Cosmos-Reason2-2B`     | Request access on the model page, then `export HF_TOKEN=...` (env var, not just `hf auth login`). |
| Demo dataset returns 512-byte / garbage tensors                  | `git lfs install && git lfs checkout` inside `third_party/isaac-gr00t`                    |
| `No space left on device` mid-`safetensors` save                 | `--output-dir` (or `TMPDIR`) is on a small `/tmp`. Move both to a large filesystem.       |
| `tyro` / `flash_attn` not found                                  | Activate the repo venv: `source third_party/isaac-gr00t/.venv/bin/activate`               |
| "no projector head found" / shape-mismatch in first forward pass | `--embodiment-tag` doesn't match the tag the modality-config Python file registers under. |
