# GR00T Data Download Guide

GR00T finetuning consumes datasets in **GR00T-flavored LeRobot v2 format** —
i.e. standard LeRobot v2 (`meta/info.json`, `meta/episodes.jsonl`,
`meta/tasks.jsonl`, chunked Parquet + MP4) **plus** an extra
`meta/modality.json` that maps state/action dimensions to named modalities.

This guide covers three sources of training data, ordered from "no work
required" to "needs conversion":

1. **Bundled demo data** — already in the repo, no download (good for
   smoke-testing).
2. **NVIDIA PhysicalAI datasets on HuggingFace** — already GR00T-flavored,
   just `snapshot_download`.
3. **Community LeRobot datasets** — usually v3 format without
   `modality.json`; need conversion + manual modality definition.

---

## Prerequisites

```bash
# Install git-lfs once (system package; needed for option 1's bundled mp4s)
sudo apt-get install -y git-lfs && git lfs install

# Authenticate to HuggingFace (some NVIDIA repos are gated — see SETUP_GUIDE.md §3)
hf auth login --token hf_xxx_your_token_here
export HF_TOKEN=hf_xxx_your_token_here

# Optional but strongly recommended: cache HF on a roomy filesystem
export HF_HOME=/path/to/large_disk/.hf_cache
```

---

## Option 1 — Bundled demo data (no download)

`isaac-gr00t` ships several small datasets under `third_party/isaac-gr00t/demo_data/`.
They arrive as 512-byte **git-lfs pointers** — you must materialise them
before they're usable:

```bash
cd third_party/isaac-gr00t
git lfs install
git lfs checkout                # ~617 MB total; turns pointers into real files

ls demo_data/
# cube_to_bowl_5             — 5 episodes, SO101 single-arm pick-place (used by examples/finetune.sh)
# cube_to_bowl_5_with_mask   — same data + segmentation masks
# droid_sample               — 1 sample episode from DROID
# libero_demo                — LIBERO single-task demo
# simplerenv_bridge_sample   — 1 sample from Bridge (SimpleEnv format)
# simplerenv_fractal_sample  — 1 sample from Fractal (SimpleEnv format)
```

`cube_to_bowl_5` is what the official `examples/finetune.sh` recipe targets.
Five episodes is enough for a 2000-step training run when combined with
color-jitter augmentation (see [SETUP_GUIDE.md §6](SETUP_GUIDE.md#6-verified-finetune-recipe-smoke-test)).

---

## Option 2 — NVIDIA GR00T-flavored datasets from HuggingFace

NVIDIA publishes several large teleop / sim datasets that already include
`meta/modality.json`. The full list:

| Repo                                                      | Size      | Embodiment             | Notes                                  |
| --------------------------------------------------------- | --------- | ---------------------- | -------------------------------------- |
| `nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1`              | ~540 MB   | Unitree G1 humanoid    | 4 fruit-pick subtasks (apple, grapes, pear, starfruit) |
| `nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1`             | ~50 GB    | Fourier GR1 humanoid   | Lab + EgoDex + DreamDojo eval splits   |
| `nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim`             | ~80 GB    | GR1 in sim             | Real-to-sim teleop trajectories        |
| `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`       | ~200 GB   | Multiple (Panda, GR1)  | 12 cross-embodiment tasks              |
| `nvidia/PhysicalAI-Robotics-GR00T-Eval`                   | small     | Mixed                  | Eval-only suite                        |

### Download the whole repo

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1",
    repo_type="dataset",
    local_dir="data/gr00t/teleop_g1",
    max_workers=8,                   # parallel file fetches
)
```

### Download a single subtask (recommended for quick experimentation)

The big repos are split by subtask folder at the top level. Use
`allow_patterns` to grab just one:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1",
    repo_type="dataset",
    allow_patterns=["g1-pick-apple/**", "README.md"],
    local_dir="data/gr00t/teleop_g1_pick_apple",
    max_workers=8,
)
# → data/gr00t/teleop_g1_pick_apple/g1-pick-apple/{data,videos,meta}/...
```

### Or via the `hf` CLI

```bash
hf download nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 \
    --repo-type dataset \
    --include "g1-pick-apple/**" "README.md" \
    --local-dir data/gr00t/teleop_g1_pick_apple
```

### Verify the dataset is GR00T-flavored

```bash
ls data/gr00t/teleop_g1_pick_apple/g1-pick-apple/meta/
# Should include: info.json  episodes.jsonl  tasks.jsonl  modality.json  stats.json
```

If `modality.json` is missing, the dataset is *plain* LeRobot — see Option 3.

### Modality config for a downloaded dataset

Each NVIDIA subtask folder has its own `meta/modality.json` describing the
state/action layout. To finetune on it you still need to write a Python
modality config that calls `register_modality_config(...)` matching those
keys (see [`finetune_new_embodiment.md`](../third_party/isaac-gr00t/getting_started/finetune_new_embodiment.md)).
For G1 humanoid data, the keys differ from SO100 — don't reuse
`examples/SO100/so100_config.py`. NVIDIA ships matching configs under
`examples/<embodiment>/` in the `isaac-gr00t` repo.

---

## Option 3 — Community LeRobot datasets (need conversion)

Most community datasets on the HF Hub (search `lerobot/`, `cadene/`,
`youliangtan/`, etc.) are published in **LeRobot v3** format, which GR00T
doesn't read natively. Two missing pieces:

1. **v3 → v2 conversion** — converts `meta/episodes/chunk-*/file-*.parquet`
   back to the older `meta/episodes.jsonl` + `meta/tasks.jsonl` layout.
2. **Add `modality.json`** — GR00T-specific metadata that maps the flat
   `observation.state` / `action` arrays to named modalities.

### Step 1: Download + convert v3 → v2

```bash
# Download the v3 dataset
hf download lerobot/svla_so101_pickplace \
    --repo-type dataset \
    --local-dir data/lerobot/svla_so101_pickplace_v3

# Run the bundled converter
cd third_party/isaac-gr00t
uv run python scripts/lerobot_conversion/convert_v3_to_v2.py \
    --input-dir  ../../data/lerobot/svla_so101_pickplace_v3 \
    --output-dir ../../data/lerobot/svla_so101_pickplace_v2
```

### Step 2: Author `meta/modality.json`

The schema is documented in [`data_preparation.md`](../third_party/isaac-gr00t/getting_started/data_preparation.md).
Minimal example for an SO100/SO101 single-arm + gripper dataset with two
cameras:

```json
{
    "state": {
        "single_arm": {"start": 0, "end": 6, "dtype": "float32"},
        "gripper":    {"start": 6, "end": 7, "dtype": "float32"}
    },
    "action": {
        "single_arm": {"start": 0, "end": 6, "dtype": "float32", "absolute": false},
        "gripper":    {"start": 6, "end": 7, "dtype": "float32", "absolute": true}
    },
    "video": {
        "front": {"original_key": "observation.images.up"},
        "wrist": {"original_key": "observation.images.side"}
    },
    "annotation": {
        "human.task_description": {"original_key": "task_index"}
    }
}
```

`start`/`end` indices must match the layout of `observation.state` and
`action` columns in the dataset's parquet files. Use the converter's
console output (it prints state/action dims) or load one parquet file with
`pandas.read_parquet` and inspect column shapes before writing.

Save as `data/lerobot/svla_so101_pickplace_v2/meta/modality.json`.

### Step 3: Validate

```bash
cd third_party/isaac-gr00t
uv run python -c "
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
import sys
sys.path.insert(0, 'examples/SO100')
import so100_config            # registers the modality config

ds = LeRobotSingleDataset(
    dataset_path='../../data/lerobot/svla_so101_pickplace_v2',
    modality_configs=...,       # see so100_config.py for the dict layout
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
print(f'Total samples: {len(ds)}')
print(f'First sample keys: {list(ds[0].keys())}')
"
```

If this prints sample shapes without error, the dataset is finetune-ready.

---

## Disk-space planning

| Item                                          | Size       |
| --------------------------------------------- | ---------- |
| `cube_to_bowl_5` (bundled, after `git lfs checkout`) | ~617 MB total demo_data |
| `g1-pick-apple` subtask (Option 2 example)    | ~140 MB    |
| Full Teleop-G1 repo                           | ~540 MB    |
| Full Teleop-GR1 / Teleop-Sim                  | 50–80 GB   |
| Full X-Embodiment-Sim                         | ~200 GB    |
| Per-finetune checkpoint (GR00T-N1.7-3B)       | ~13 GB     |

For anything beyond Option 1, point your downloads at a filesystem with
**at least 50 GB free** for the dataset + 50 GB for the cache + 30 GB for
two checkpoints. See [SETUP_GUIDE.md §5](SETUP_GUIDE.md#5-disk-space) for
why `/tmp` is usually the wrong choice.
