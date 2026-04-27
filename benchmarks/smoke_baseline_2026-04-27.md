# RoboRenForce VLA-RL Smoke Baseline — 2026-04-27

Frozen Qwen2-VL-2B-Instruct policy, 1 GRPO iteration, group_size=8.
Captured before any pretraining / SFT, so all metrics are at-init.

## Results

| Task | num_envs | iter | return | success | loss | kl | wallclock | status |
|---|---|---|---|---|---|---|---|---|
| LIBERO-Spatial-GRPO-v0 | 4 | 1/1 | 0.000 | 100.00% | 0.0000 | 0.0009 | 560.1s | ✅ |
| CALVIN-D-GRPO-v0 | 2 | 1/1 | 0.000 | 100.00% | 0.0000 | 0.0002 | 577.3s | ✅ |
| ManiSkill-PickCube-GRPO-v0 (state) | 8 | 1/1 | — | — | — | — | 93s | ✅ prior run |
| ManiSkill-PickCube-GRPO-v0 (rgbd) | — | — | — | — | — | — | — | ⚠ blocked: SAPIEN render hangs in `nvidia-eglcore::poll` post Vulkan install |
| RoboTwin-PlaceCup-GRPO-v0 | — | — | — | — | — | — | — | ⚠ blocked: same SAPIEN render hang |

## Caveats

- `return=0.000` because Qwen2-VL is frozen and outputs a zero-mean continuous head; episodes never see task reward.
- `success=100%` here means *episode completed without crash* (auto-reset path engaged), not task-solved. Treat as a liveness check, not a policy quality metric.
- `loss=0.0` with `kl≈1e-3` is expected for one GRPO step on a frozen policy (no advantage signal).
- LIBERO/CALVIN wallclock dominated by Qwen2-VL forward through max_episode_steps × num_envs, not by sim physics.

## Reproduce

```
cd projects/RoboRenForce
./.venv/bin/python scripts/vla/rl/train_libero.py  --task LIBERO-Spatial-GRPO-v0 --num_envs 4 --max_iterations 1 --logdir benchmarks/_run_libero
./.venv/bin/python scripts/vla/rl/train_calvin.py  --task CALVIN-D-GRPO-v0       --num_envs 2 --max_iterations 1 --logdir benchmarks/_run_calvin
./.venv/bin/python scripts/vla/rl/train_maniskill.py --task ManiSkill-PickCube-GRPO-v0 --num_envs 8 --max_iterations 1 --obs_mode state --logdir benchmarks/_run_maniskill
```

## Environment

- Host: 8× NVIDIA H100 80GB HBM3, driver 535.129.03
- venv: `projects/RoboRenForce/.venv` (Python 3.10, torch 2.7.0+cu126)
- Vulkan loader installed today (`libvulkan1` + `vulkan-tools` + `mesa-vulkan-drivers`); `vulkaninfo --summary` lists all 8 H100 OK
- SAPIEN 3.0.3 render still hangs in `cam.get_picture` → `libnvidia-eglcore.so::poll` despite ICD/loader in place; deferred for separate investigation
