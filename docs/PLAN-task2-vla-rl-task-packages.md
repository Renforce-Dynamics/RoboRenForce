# Task 2 Plan — VLA RL Task Packages + Per-Task Scripts

> **Goal**: Remove every mock from VLA RL training. Mirror the locomotion training pattern (`train_mjlab.py` + `RRF_mjlab` task package + gym-registered task IDs) so that `python scripts/vla/rl/train_<benchmark>.py --task <Task-Algo-ID>` resolves a complete (env, actor, algorithm, runner) bundle from the task registry. No algorithm name in script names. No cross-benchmark imports in any script.

---

## 1. Why this layout

Existing pattern (locomotion, `scripts/renforce/train_mjlab.py:72-91`):

```python
task_id = "Mjlab-Velocity-Flat-Unitree-Go1-PPO"
spec = gym.spec(task_id)
runner_cfg = spec.kwargs["RoboRenForce_entry_point"]   # MJLabLocoPPOCfg()
env_cfg   = spec.kwargs["env_cfg_entry_point"]         # MJLabFlatGo1EnvCfg()
runner    = runner_cfg.class_type(runner_cfg, env=env_cfg.build(), ...)
runner.learn()
```

Algorithm choice (PPO/SAC/GRPO) lives **inside** the registered task ID and the bundled `runner_cfg`. The script is a thin launcher.

We replicate this verbatim for VLA RL.

---

## 2. Target file tree

```
scripts/vla/rl/
├── _common.py                # CLI parser, runner build helper. Imports ONLY core.
├── train_robotwin.py         # Imports RRF_robotwin_vla_rl ONLY
├── train_libero.py           # Imports RRF_libero_vla_rl ONLY
├── train_maniskill.py        # Imports RRF_maniskill_vla_rl ONLY
├── train_calvin.py           # Imports RRF_calvin_vla_rl ONLY
└── train_d4rl.py             # Imports RRF_d4rl_vla_rl ONLY

source/tasks/
├── RRF_robotwin_vla_rl/                          # NEW
│   ├── pyproject.toml                            # depends on RRF_robotwin + RoboRenForce-core
│   └── RRF_robotwin_vla_rl_tasks/
│       ├── __init__.py                           # gym.register all RoboTwin VLA-RL task IDs
│       ├── _registry.py                          # helper: register(task_id, env_cfg, runner_cfg)
│       ├── place_empty_cup/
│       │   ├── __init__.py
│       │   ├── env_cfg.py                        # RoboTwinPlaceCupEnvCfg
│       │   ├── agents_grpo.py                    # RoboTwinPlaceCupGRPOCfg
│       │   ├── agents_ppo.py                     # RoboTwinPlaceCupPPOCfg
│       │   └── actor.py                          # RoboTwinPlaceCupVLAActorCfg
│       ├── stack_blocks/
│       │   └── ... (same shape)
│       └── _shared/
│           ├── vla_actor_presets.py              # Qwen2VLActorCfg / GR00TActorCfg presets
│           └── algo_presets.py                   # GRPOCfg / PPOCfg defaults tuned for RoboTwin
│
├── RRF_libero_vla_rl/        # NEW (same shape)
├── RRF_maniskill_vla_rl/     # NEW
├── RRF_calvin_vla_rl/        # NEW
└── RRF_d4rl_vla_rl/          # NEW
```

**Why `RRF_<benchmark>_vla_rl` instead of merging into `RRF_<benchmark>`**:
- `RRF_robotwin` already exists for SL/replay use; we do NOT want to force a `RoboRenForce-core` dependency on users who only want the simulator wrapper.
- `RRF_<benchmark>_vla_rl` is opt-in: it pulls in `RoboRenForce-core[vla]` + benchmark sim deps.

---

## 3. Naming convention for task IDs

```
<Benchmark>-<Task>-<Algorithm>-v<n>
```

Examples:
- `RoboTwin-PlaceCup-GRPO-v0`
- `RoboTwin-PlaceCup-PPO-v0`
- `LIBERO-Spatial-GRPO-v0`
- `LIBERO-Object-PPO-v0`
- `ManiSkill-PickCube-GRPO-v0`
- `CALVIN-D-PPO-v0`
- `D4RL-Hopper-SAC-v0`

The algorithm suffix is the **only** way to switch algorithms. There is no `--algorithm` CLI flag. This matches `Mjlab-Velocity-Flat-Unitree-Go1-PPO` vs `…-SAC` exactly.

---

## 4. The script template

`scripts/vla/rl/_common.py`:

```python
"""Shared CLI parsing + runner construction. MUST NOT import any RRF_<benchmark>_* package."""
import argparse
import gymnasium as gym
from RoboRenForce.utils.argtool import build_runner_from_task_spec

def make_parser(benchmark_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"VLA RL training on {benchmark_name}")
    p.add_argument("--task", required=True, help=f"Registered {benchmark_name} VLA-RL task ID")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_iterations", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--logdir", default=None)
    return p

def run(args):
    spec = gym.spec(args.task)
    env_cfg    = spec.kwargs["env_cfg_entry_point"]
    runner_cfg = spec.kwargs["RoboRenForce_entry_point"]
    if args.num_envs:       env_cfg.num_envs = args.num_envs
    if args.max_iterations: runner_cfg.max_iterations = args.max_iterations
    if args.seed:           runner_cfg.seed = args.seed
    if args.logdir:         runner_cfg.logger_cfg.log_dir = args.logdir

    runner = build_runner_from_task_spec(runner_cfg, env_cfg, device=args.device)
    runner.learn()
```

`scripts/vla/rl/train_robotwin.py`:

```python
"""Train any RoboTwin-* VLA-RL task. Engine init for sapien lives here."""
import sys
# 1. Engine init (sapien renderer must be imported BEFORE torch on some setups)
import sapien  # noqa
import torch   # noqa

# 2. Import the task package — registers all RoboTwin-*-{GRPO,PPO}-v0 task IDs
import RRF_robotwin_vla_rl_tasks  # noqa

from _common import make_parser, run

if __name__ == "__main__":
    parser = make_parser("RoboTwin")
    args = parser.parse_args()
    sys.exit(run(args))
```

Other 4 scripts identical except for engine init + import line.

---

## 5. Task registration template

`source/tasks/RRF_robotwin_vla_rl/RRF_robotwin_vla_rl_tasks/__init__.py`:

```python
from ._registry import register_robotwin_task

# place_empty_cup
from .place_empty_cup.env_cfg     import RoboTwinPlaceCupEnvCfg
from .place_empty_cup.agents_grpo import RoboTwinPlaceCupGRPOCfg
from .place_empty_cup.agents_ppo  import RoboTwinPlaceCupPPOCfg

register_robotwin_task("RoboTwin-PlaceCup-GRPO-v0", RoboTwinPlaceCupEnvCfg, RoboTwinPlaceCupGRPOCfg)
register_robotwin_task("RoboTwin-PlaceCup-PPO-v0",  RoboTwinPlaceCupEnvCfg, RoboTwinPlaceCupPPOCfg)

# stack_blocks
from .stack_blocks.env_cfg     import RoboTwinStackBlocksEnvCfg
from .stack_blocks.agents_grpo import RoboTwinStackBlocksGRPOCfg
register_robotwin_task("RoboTwin-StackBlocks-GRPO-v0", RoboTwinStackBlocksEnvCfg, RoboTwinStackBlocksGRPOCfg)
```

`_registry.py`:

```python
import gymnasium as gym

def register_robotwin_task(task_id: str, env_cfg_cls, runner_cfg_cls):
    gym.envs.registry.pop(task_id, None)  # idempotent
    gym.register(
        id=task_id,
        entry_point=lambda *a, **kw: env_cfg_cls().build(),
        kwargs={
            "env_cfg_entry_point":      env_cfg_cls(),
            "RoboRenForce_entry_point": runner_cfg_cls(),
        },
    )
```

`place_empty_cup/agents_grpo.py`:

```python
from RoboRenForce.utils.configclass import configclass
from RoboRenForce import runners, components, algorithms
from .._shared.vla_actor_presets import qwen2vl_actor_preset
from .._shared.algo_presets       import grpo_robotwin_default
from .actor import RoboTwinPlaceCupVLAActorCfg

@configclass
class RoboTwinPlaceCupGRPOCfg(runners.VLAGRPORunnerCfg):
    seed             = 42
    num_steps_per_env = 64
    max_iterations   = 1000

    policy = components.ActorCriticPackCfg(
        actor_cfg=RoboTwinPlaceCupVLAActorCfg(
            vlm_cfg=qwen2vl_actor_preset(),
        ),
    )
    algorithm  = grpo_robotwin_default()  # GRPOCfg with KL penalty 0.02, lora_r=8, etc.
    logger_cfg = runners.AgentLoggerCfg(enable_writer=True)
```

---

## 6. Step-by-step implementation

### Step 0 — Scaffold (45 min)

- [ ] Create the 5 `RRF_<benchmark>_vla_rl/` directories with `pyproject.toml` (depends on the matching `RRF_<benchmark>` + `RoboRenForce-core`).
- [ ] Create the 5 script files in `scripts/vla/rl/` plus `_common.py`.
- [ ] Add `build_runner_from_task_spec` helper in `RoboRenForce/utils/argtool.py` (mirror of `parse_rl_cfg` for VLA case).
- [ ] Smoke test: `python -c "import RRF_robotwin_vla_rl_tasks; import gymnasium as gym; print(gym.spec('RoboTwin-PlaceCup-GRPO-v0'))"`.

### Step 1 — RoboTwin (first benchmark to wire) (2–3 h)

- [ ] Read `RRF_robotwin/RRF_robotwin_tasks/envs/robotwin_env.py` to understand env API.
- [ ] Implement `RoboTwinPlaceCupEnvCfg` wrapping `RoboTwinEnv(task_name="place_empty_cup")` with the multimodal env wrapper (image_mode="render", language="task_description", state="proprio").
- [ ] Implement `RoboTwinPlaceCupVLAActorCfg`:
  - `vlm_cfg = Qwen2VLCfg(model_path=…, lora_r=8)`
  - `action_head_cfg = DiffusionActionHeadCfg(action_dim=14, chunk_size=16)`  (RoboTwin dual-arm = 14 dim)
- [ ] Implement `RoboTwinPlaceCupGRPOCfg` (above template).
- [ ] Run: `python scripts/vla/rl/train_robotwin.py --task RoboTwin-PlaceCup-GRPO-v0 --num_envs 4 --max_iterations 10`
- [ ] **Acceptance**: 10 iterations complete without crash, `events.jsonl` shows decreasing `policy_loss`, episode length > 1.

### Step 2 — Delete mocks from old scripts (30 min)

- [ ] `scripts/vla/rl/train_robotwin_grpo.py` → delete (replaced by `train_robotwin.py`).
- [ ] `scripts/vla/rl/train_vla_benchmark.py` → delete (replaced by 4 per-benchmark scripts).
- [ ] In `multimodal_env_wrapper.py`, change `image_mode: str = "dummy"` → `"render"` (default). Keep "dummy" as opt-in for unit tests.
- [ ] Verify `tests/test_grpo.py:MockEmbodiedEnv` is **kept** (it's a test fixture, not user-facing).

### Step 3 — LIBERO (2 h)

- [ ] Mirror Step 1 structure: `RRF_libero_vla_rl_tasks/spatial_pick_object/{env_cfg,agents_grpo,actor}.py`.
- [ ] Initial task IDs: `LIBERO-Spatial-GRPO-v0`, `LIBERO-Object-GRPO-v0`, `LIBERO-Goal-GRPO-v0`, `LIBERO-Long-GRPO-v0`.
- [ ] Smoke test 10 iter on `LIBERO-Spatial-GRPO-v0`.

### Step 4 — ManiSkill / CALVIN / D4RL (2 h each, parallel)

- [ ] Repeat Step 3 pattern. For D4RL use SAC suffix instead of GRPO (off-policy fits offline-to-online better).

### Step 5 — Smoke test suite (1 h)

- [ ] `tests/test_runners/test_vla_rl_smoke.py`: parametrize over all 5 benchmarks, run 5 iter each on smallest task ID per benchmark, assert no crash.
- [ ] CI marker `@pytest.mark.vla_rl_smoke` so this can be skipped on machines without all sims installed.

### Step 6 — Documentation (30 min)

- [ ] Update README "Quick Start" with one example per benchmark.
- [ ] Update `docs/SETUP_GUIDE.md` to mention `RRF_<benchmark>_vla_rl` as the install for VLA RL.

---

## 7. Acceptance criteria (whole task)

| | Criterion |
|---|---|
| ✅ | `scripts/vla/rl/` contains exactly 5 per-benchmark scripts + `_common.py`. No script imports more than one `RRF_<benchmark>_*` package. |
| ✅ | No file under `scripts/` or `source/RoboRenForce/` contains `Mock`/`Dummy`/`Fake` related to env/policy (test fixtures excepted). |
| ✅ | Each of the 5 benchmarks has at least one task ID registered for both GRPO and PPO. |
| ✅ | `python scripts/vla/rl/train_robotwin.py --task RoboTwin-PlaceCup-GRPO-v0 --num_envs 2 --max_iterations 5` runs end-to-end. |
| ✅ | `pytest -m vla_rl_smoke tests/test_runners/test_vla_rl_smoke.py` passes (or skips cleanly with installation hint when a sim is missing). |
| ✅ | `multimodal_env_wrapper.image_mode` default is `"render"`, not `"dummy"`. |

---

## 8. Risks & mitigation

| Risk | Mitigation |
|---|---|
| RoboTwin env wrapper API drift since last sync | Step 1 begins with reading current `RoboTwinEnv` source, not assuming. |
| GRPO config tuned for mocks doesn't converge on real obs | Algo presets in `_shared/algo_presets.py` are starting points; first real run may need lr / KL retuning — that's expected, not a failure of this plan. |
| `gym.register` with same ID running twice raises | `_registry.py` does `gym.envs.registry.pop(task_id, None)` first (idempotent). |
| sapien import order issues with torch | `train_robotwin.py` imports sapien before torch (template enforces). |
