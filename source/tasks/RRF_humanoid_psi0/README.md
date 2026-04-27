# RRF_humanoid_psi0_tasks

Psi0 (USC Physical Superintelligence Lab) humanoid datasets and robot configs.

- Robot: **Unitree G1 / Dex3** (whole-body teleop)
- Format: **LeRobot v2** (per-episode parquet + mp4)
- Source: <https://github.com/physical-superintelligence-lab/Psi0>
- Hub: <https://huggingface.co/datasets/USC-PSI-Lab/psi-data>

## Dataset download

Use the helper script at `scripts/data/download_psi0_dataset.py`. It calls
`hf download` (HuggingFace CLI) under the hood and unzips into a LeRobot v2
directory tree.

```bash
# List available tasks
python scripts/data/download_psi0_dataset.py --list

# Smallest sim task (~30MB) — smoke tests
python scripts/data/download_psi0_dataset.py \
    --task G1WholebodyTabletopGraspMP-v0 --split simple \
    --output-dir /vepfs/users/zza/psi0_data

# Smallest real task (~274MB)
python scripts/data/download_psi0_dataset.py \
    --task Hold_lunch_bag_with_both_hands_and_squat_to_put_on_the_coffee_table \
    --split real --output-dir /vepfs/users/zza/psi0_data

# All sim tasks (~500MB)
python scripts/data/download_psi0_dataset.py --task all --split simple \
    --output-dir /vepfs/users/zza/psi0_data
```

After unzip, the layout for one task is:

```
<output-dir>/<split>/<task_name>/
  meta/
    info.json, episodes.jsonl, tasks.jsonl, stats.json, modality.json
  data/chunk-000/episode_<NNNNNN>.parquet
  videos/chunk-000/<view>/episode_<NNNNNN>.mp4
```

### Authentication

The hub is **public** (no token required). If you see rate-limit warnings,
run `hf auth login` once to set `HF_TOKEN`.

## Training

Pretrain (single GPU, mock VLM, real Psi0 data):

```bash
python scripts/vla/pretrain/train_single_gpu.py --psi0 \
    --data_root /vepfs/users/zza/psi0_data/simple/G1WholebodyTabletopGraspMP-v0 \
    --epochs 1 --batch_size 4 --device cuda
```

Switch to a real backbone (Qwen2-VL / GR00T / OpenPI) by adapting
`build_psi0_mock_config()` in `scripts/vla/pretrain/train_single_gpu.py`
to use the corresponding `Q*VL` / `GR00T*` / `OpenPI*` policy config.

## Embodiments

The robot config is defined in `RRF_humanoid_psi0_tasks/robots/UnitreeG1Cfg.py`:

- `action_dim = 36` (standardized; raw 28D auto-promoted)
- `state_dim  = 32`
- Default cameras: head egocentric

## Other tasks (HE, EgoDex, raw teleop)

`USC-PSI-Lab/psi-data` also ships:

- `HE_RAW.zip` (230 GB), `HE_RAW_no_static.zip` (176 GB) — Humanoid Everyday
- `egodex-sample.zip` (59 MB), `egodex_retargeting.zip` (5.65 GB) — EgoDex retargeting
- `egoverse-sample.zip` (1.09 GB)
- `simple-teleop/` — raw teleoperation traces

Download these directly with `hf download`:

```bash
hf download USC-PSI-Lab/psi-data egodex-sample.zip \
    --repo-type dataset --local-dir /vepfs/users/zza/psi0_data
```

Pre-trained Psi0 policy checkpoints are at <https://huggingface.co/USC-PSI-Lab/psi-model>.
