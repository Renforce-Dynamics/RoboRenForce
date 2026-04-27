# RRF_orchestra

Process-level parallelism for VLA RL training. Lives outside `RoboRenForce`
core so single-process users do not pay for `multiprocessing` orchestration.

## What this package is (and is not)

There are **two orthogonal axes of distribution** in this repo. Keep them
separate in your head — they solve different problems.

| Axis | Where it lives | Problem it solves | When you need it |
|---|---|---|---|
| **Role-split (this package)** | `RRF_orchestra/` | Sim CPU and inference GPU are idle waiting on each other. One process per *role*: `EnvWorker × N`, `InferenceWorker × 1`, `Learner × 1`. Tensors moved over `torch.multiprocessing.Queue` with shared-memory zero-copy. | VLA RL on a **single machine**: VLA model dominates step time; you want batched inference across N parallel envs. |
| **Data-parallel / DDP** | `RoboRenForce/runners/vla/{pretrain,post_train}/*_runner_distributed.py` + `scripts/vla/{pretrain,post_train}/train_*_ddp.py` | Single GPU is too small / too slow for a big batch. **Same role on every rank**, gradients all-reduced via NCCL. Launched with `torchrun --nproc_per_node=8`. | VLA **pretrain / SFT** on N GPUs (one machine or many) where each rank does a full forward+backward on its slice of the batch. |

You can stack them — e.g. role-split rollout + DDP learner — but each axis
is independently useful and the v1 implementation here keeps them decoupled:
the `Learner` worker in this package is single-process today; switching it
to a DDP learner is a swap-in (`policy_factory` returns `DDP(model)`).

## Architecture (role-split)

```
   EnvWorker × N           InferenceWorker × 1         Learner × 1
   ┌───────────┐  ObsBatch  ┌─────────────────┐         ┌─────────┐
   │ env.step  │ ─────────▶ │ batched VLA inf │ ──────▶ │ collect │
   │           │ ◀───────── │ ActionBatch     │         │ traj    │
   └─────┬─────┘            └────────▲────────┘         │ grad    │
         │ Trajectory                 │ WeightUpdate    │ update  │
         ▼                            └─────────────────┤         │
   ┌──────────────────────────────────────────────────────┘
   │              channels (mp.Queue + shared mem tensors)
   └────────────────────────────────────────────────────────
```

All large tensors travel via `SharedTensorRef` (zero-copy across processes).
Only tiny handles + metadata go through the queue.

## Status

- [x] Step 0: package scaffolding
- [x] Step 1: wire protocol (`shared_tensor`, `messages`, `channels`) — 13 tests
- [x] Step 2: `BaseEnvWorker`, `InferenceWorker`, `FixedBatcher` / `DynamicBatcher` — 3 tests
- [x] Step 3: orchestrator (`Topology`, `OrchestraVLARunner`, `Supervisor`, `weight_sync`) — 4 tests
- [x] Step 3.5: adapters (`PolicyAdapter`, `AlgorithmAdapter`)
- [x] Step 3.6: end-to-end smoke (`examples/hello_world_orchestra.py`)
- [ ] Step 4: per-task `EnvWorker` for first benchmark (RoboTwin / LIBERO) — task-package responsibility
- [ ] Step 5: throughput benchmark (`examples/benchmark_throughput.py`)
- [ ] Step 6: failure-handling integration tests (worker kill mid-run)

## Install

```bash
pip install -e source/RRF_orchestra
```

## Quick test

```bash
# 20 unit/integration tests (mp + shared-memory + orchestrator round-trip)
pytest tests/RRF_orchestra/ -v

# Full hello-world: 2 env workers + 1 inference + 1 learner, 5 iterations
python source/RRF_orchestra/RRF_orchestra/examples/hello_world_orchestra.py
```

Expected hello-world tail:
```
[iter  4] {'iter': 4, 'num_trajs': 2, 'iter_calls': 5}
finished — collected 5 iteration(s)
```

## Wire protocol

The single source of truth for cross-process messages is
`RRF_orchestra/protocol/messages.py`. Five message types, all
`@configclass`, all carry `schema_version=PROTOCOL_VERSION`. Any breaking
schema change bumps `PROTOCOL_VERSION` and workers refuse to start on
mismatch (`assert_compatible(msg)` in `Channel.get`).

| Message | Producer | Consumer | Channel |
|---|---|---|---|
| `ObsBatch` | EnvWorker | InferenceWorker | `obs_ch` |
| `ActionBatch` | InferenceWorker | EnvWorker `i` | `action_ch[i]` |
| `Trajectory` | EnvWorker | Learner | `traj_ch` |
| `WeightUpdate` | Learner | InferenceWorker | `weight_ch` |
| `ControlMsg` | Supervisor ↔ Worker | bidirectional | `ctrl_ch[i]` |

See `docs/PLAN-task3-orchestra-package.md` for the full design rationale.

## Adding a per-task `EnvWorker`

Concrete env workers live **inside their own task package**, not here.
Subclass `BaseEnvWorker` and implement four methods:

```python
from RRF_orchestra.workers.env_worker import BaseEnvWorker, BaseEnvWorkerCfg

class RoboTwinPlaceCupEnvWorker(BaseEnvWorker):
    def setup(self):       ...   # init sapien INSIDE the worker process
    def reset_envs(self, env_ids):  ...  # → ObsBatch
    def step_envs(self, action):    ...  # → (next_obs, reward, done, info)
    def teardown(self):    ...
```

The base class provides the run loop, signal handling, channel I/O, and
trajectory emission.
