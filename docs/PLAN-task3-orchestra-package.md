# Task 3 Plan — `RRF_orchestra` Package: Process-Level Parallelism for VLA RL

> **Goal**: Solve the "single process is wasted GPU↔CPU ping-pong" problem for VLA RL training. One process holds the (huge) VLA model and serves batched inference. N processes step independent envs. The learner pulls trajectories and pushes weights back. **All of this lives in a separate package** (`RRF_orchestra`) that depends on the core repo but never modifies it. Per-task env workers are implemented inside their `RRF_<benchmark>_vla_rl` task package via a strict abstract prototype + a fully-defined wire protocol.

---

## 1. The problem we are solving

In single-process VLA RL today:

```
  while True:
      obs  = env.step(prev_action)      # CPU sim — GPU idle
      act  = vla_policy(obs)             # GPU infer — CPU idle
      buf.add(obs, act, ...)
```

VLA models (Qwen2-VL 2B, GR00T 3B) make inference the dominant cost. The fix is to **separate the roles into processes**:

```
   Sim Worker × N         Inference Worker (1)        Learner (1)
   ┌───────────┐  obs    ┌─────────────────┐          ┌─────────┐
   │  env.step │ ─────▶  │ batched VLA infer│ ──────▶  │ collect │
   │           │ ◀───── action               │          │ traj    │
   └─────┬─────┘         └────────▲────────┘          │ grad    │
         │ trajectory              │ weight push       │ update  │
         ▼                         └───────────────────┤         │
   ┌───────────────────────────────────────────────────┘         │
   │                  TrajectoryChannel                          │
   └─────────────────────────────────────────────────────────────┘
```

**Why a separate package**: keeps `RoboRenForce-core` light. Users running single-process locomotion never pull `multiprocessing` orchestration code. Orchestra-mode users opt in by `pip install -e source/RRF_orchestra`.

**Why `torch.multiprocessing.Queue` (not Ray, not gRPC)**:
- 0 new heavy dependencies (torch already imported)
- Single-machine first-class; tensors share memory zero-copy via `tensor.share_memory_()`
- `pdb.set_trace()` works in workers
- Multi-machine is explicitly a v2 concern; the protocol is designed to allow swapping the channel layer later without touching the worker code.

---

## 2. Target file tree (NEW package only)

```
source/RRF_orchestra/                              # NEW package
├── pyproject.toml                                   # depends on RoboRenForce-core
├── README.md                                        # 1-page user guide
└── RRF_orchestra/
    ├── __init__.py
    │
    ├── protocol/                                    # The wire contract — strict, versioned
    │   ├── __init__.py
    │   ├── messages.py                              # ObsBatch, ActionBatch, Trajectory, WeightUpdate, ControlMsg
    │   ├── shared_tensor.py                         # SharedTensorRef: zero-copy tensor handle across processes
    │   ├── codec.py                                 # pack / unpack message → mp.Queue payload
    │   └── channels.py                              # ObsChannel, ActionChannel, TrajChannel, WeightChannel, ControlChannel
    │
    ├── workers/                                     # Process bodies + abstract prototypes
    │   ├── __init__.py
    │   ├── base_worker.py                           # BaseWorker: run loop, lifecycle, signal handling
    │   ├── env_worker.py                            # BaseEnvWorker (abstract); EnvWorkerCfg
    │   ├── inference_worker.py                      # InferenceWorker; InferenceWorkerCfg
    │   ├── learner_worker.py                        # LearnerWorker (in-process by default)
    │   └── batcher.py                               # FixedBatcher / DynamicBatcher used by InferenceWorker
    │
    ├── orchestrator/                                # Spawn + supervise + shut down
    │   ├── __init__.py
    │   ├── topology.py                              # Topology: declares N env workers, 1 inference, 1 learner
    │   ├── orchestra_runner.py                    # OrchestraVLARunner (subclass of BaseRunnerCfg)
    │   ├── weight_sync.py                           # push/pull weights via shared memory
    │   └── supervisor.py                            # health check, restart-on-crash, graceful shutdown
    │
    ├── adapters/                                    # Glue from core single-process objects → orchestra
    │   ├── __init__.py
    │   ├── algorithm_adapter.py                     # Wraps core Algorithm to consume Trajectory msgs
    │   └── policy_adapter.py                        # Wraps core VLAActor for inference-worker use (no grad, batched forward)
    │
    └── examples/
        ├── train_robotwin_orchestra.py            # End-to-end orchestra example
        └── benchmark_throughput.py                  # Compare single-proc vs orchestra steps/sec
```

Per-task env workers live inside their **own** `RRF_<benchmark>_vla_rl` package:

```
source/tasks/RRF_robotwin_vla_rl/RRF_robotwin_vla_rl_tasks/
├── place_empty_cup/
│   ├── env_worker.py        # NEW: class RoboTwinPlaceCupEnvWorker(BaseEnvWorker)
│   ...
```

This is the per-task customization the user asked for.

---

## 3. The wire protocol — what the user explicitly demanded

> 我们对应好的内容是需要完全定义好统一的接口，和数据链路的协议的（怎么传，传什么）

This section is the **single source of truth** for cross-process communication. Every change to it requires bumping `PROTOCOL_VERSION` in `protocol/messages.py`.

### 3.1 Message types (`protocol/messages.py`)

```python
PROTOCOL_VERSION = 1

@configclass
class ObsBatch:
    """SimWorker → InferenceWorker. One per env step (or one per N envs in fixed-batch mode)."""
    worker_ids: list[int]                  # which sim workers contributed
    env_ids:    list[int]                  # which env-instance within each worker
    step_ids:   list[int]                  # monotonically increasing per (worker, env)
    images:     SharedTensorRef            # uint8 (B, num_cams, C, H, W); shared mem
    states:     SharedTensorRef            # float32 (B, state_dim); shared mem
    languages:  list[str] | SharedTensorRef  # tokenized ids OR raw strings (cheap to send)
    timestamp:  float                      # producer wall clock
    schema_version: int = PROTOCOL_VERSION

@configclass
class ActionBatch:
    """InferenceWorker → SimWorker. Mirrors ObsBatch indexing."""
    worker_ids: list[int]
    env_ids:    list[int]
    step_ids:   list[int]
    actions:    SharedTensorRef            # float32 (B, chunk_size, action_dim)
    log_probs:  SharedTensorRef | None     # float32 (B,) — required for on-policy
    weight_version: int                    # which model version produced these actions
    timestamp:  float
    schema_version: int = PROTOCOL_VERSION

@configclass
class Trajectory:
    """SimWorker → Learner. One per finished rollout segment of length T."""
    worker_id: int
    env_id:    int
    obs:        list[ObsBatch]             # length T (per-step ObsBatches with batch=1)
    actions:    SharedTensorRef            # (T, chunk_size, action_dim)
    rewards:    SharedTensorRef            # (T,)
    dones:      SharedTensorRef            # (T,) bool
    values:     SharedTensorRef | None     # (T,) — if critic in inference worker
    log_probs:  SharedTensorRef | None     # (T,)
    info:       dict                        # episode_return, success, etc.
    weight_version_range: tuple[int, int]  # min, max version used during this segment
    schema_version: int = PROTOCOL_VERSION

@configclass
class WeightUpdate:
    """Learner → InferenceWorker. After each gradient step (or every K steps)."""
    weight_version: int
    state_dict_handles: dict[str, SharedTensorRef]   # {param_name: shared tensor}
    is_full: bool                          # True = full state dict; False = LoRA delta only
    timestamp: float
    schema_version: int = PROTOCOL_VERSION

@configclass
class ControlMsg:
    """Bidirectional. Lifecycle and supervision."""
    kind: Literal["start", "stop", "pause", "health_ping", "health_pong", "fatal_error"]
    sender: str                            # worker name
    payload: dict | None
    timestamp: float
    schema_version: int = PROTOCOL_VERSION
```

### 3.2 Channels (`protocol/channels.py`)

Every channel is a **thin wrapper** around `mp.Queue` with type checking.

```python
class Channel(Generic[M]):
    def __init__(self, msg_type: type[M], maxsize: int):
        self._q = torch.multiprocessing.Queue(maxsize=maxsize)
        self._msg_type = msg_type

    def put(self, msg: M, timeout: float | None = None) -> None: ...
    def get(self, timeout: float | None = None) -> M: ...
    def get_batch(self, max_batch: int, timeout_s: float) -> list[M]: ...   # InferenceWorker uses this

# Concrete channel types (semantic naming, same impl)
ObsChannel       = Channel[ObsBatch]
ActionChannel    = Channel[ActionBatch]
TrajChannel      = Channel[Trajectory]
WeightChannel    = Channel[WeightUpdate]
ControlChannel   = Channel[ControlMsg]
```

### 3.3 Channel topology (which channel connects what)

| Channel | Producer | Consumer | maxsize | Why |
|---|---|---|---|---|
| `obs_ch`     | SimWorker × N | InferenceWorker × 1 | 4 × N | Backpressure on sim side if infer slow |
| `action_ch[i]` | InferenceWorker | SimWorker `i` (each gets its own) | 4 | Per-sim-worker queue avoids head-of-line blocking |
| `traj_ch`    | SimWorker × N | Learner | 2 × N | Backpressure on sim side if learner slow |
| `weight_ch`  | Learner | InferenceWorker | 2 | Most recent weights wins; old discarded |
| `ctrl_ch[i]` | Supervisor ↔ Worker `i` | bidirectional | 8 | Lifecycle |

### 3.4 Tensor transport: `SharedTensorRef`

```python
@dataclass
class SharedTensorRef:
    """Reference to a tensor allocated in shared memory. Copy-free across processes."""
    storage_handle: bytes        # torch.multiprocessing rebuild handle
    shape: tuple[int, ...]
    dtype: torch.dtype

    def materialize(self) -> torch.Tensor:
        """Reconstruct the tensor in the current process. Zero-copy."""
        ...

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "SharedTensorRef":
        t.share_memory_()
        return cls(...)
```

All large tensors (images, action chunks, weight tensors) MUST be passed via `SharedTensorRef`. The mp.Queue payload only carries the tiny handle, not the bytes.

### 3.5 Versioning + drift policy

- `PROTOCOL_VERSION` bumped on any breaking schema change. Workers refuse to start if their version differs from the orchestrator.
- `weight_version` monotonic. SimWorker stamps every Trajectory with the min/max version of weights its actions came from. Learner can choose to discard trajectories with too-old weights (off-policy correction is the user's choice via `algorithm.max_weight_lag`).
- `step_ids` are SimWorker-local; never reordered globally.

### 3.6 Failure semantics

- Worker crash → Supervisor sees missing `health_pong` within `health_timeout_s` → supervisor logs the cause and shuts down the whole topology. v1: no auto-restart (keeps debugging simple). v2: opt-in restart for SimWorker only (InferenceWorker / Learner restart requires checkpoint — out of scope).
- Channel `put` timeout → SimWorker drops the obs and emits a `dropped_obs` metric.
- Channel `get` timeout in InferenceWorker → it does not block forever; it just waits for the next batch.

---

## 4. The abstract `BaseEnvWorker` (per-task customization point)

`workers/env_worker.py`:

```python
class BaseEnvWorker(BaseWorker):
    """Abstract prototype. One subclass per (benchmark, task) lives in RRF_<benchmark>_vla_rl."""

    cfg: "BaseEnvWorkerCfg"

    # ---- Subclasses MUST implement ----

    def setup(self) -> None:
        """Initialize simulator inside the worker process (after fork/spawn).
        Per task: load scene, init renderer, etc. NEVER call before spawn — sim
        engines like sapien must be initialized in the worker process."""

    def reset_envs(self, env_ids: list[int]) -> ObsBatch:
        """Reset given env instances and return their initial observations."""

    def step_envs(self, action: ActionBatch) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, list[dict]]:
        """Step given envs with the action chunk. Returns (next_obs, reward, done, info)."""

    def teardown(self) -> None:
        """Release simulator resources."""

    # ---- Provided by base ----

    def run(self) -> None:
        """Main loop. SimWorker pseudocode:
            self.setup()
            obs = self.reset_envs(all_env_ids)
            while not self.should_stop():
                self.obs_ch.put(obs)
                action = self.action_ch.get()                # blocks
                next_obs, reward, done, info = self.step_envs(action)
                if done.any(): handle_episode_end(...)
                self.maybe_emit_trajectory(...)
                obs = next_obs
            self.teardown()
        """
```

`@configclass BaseEnvWorkerCfg(ModuleBaseCfg)` carries `num_envs_per_worker`, `chunk_size`, `obs_ch`, `action_ch`, `traj_ch`, `ctrl_ch` injected by the Topology.

### 4.1 Example concrete worker (lives in task package)

`source/tasks/RRF_robotwin_vla_rl/RRF_robotwin_vla_rl_tasks/place_empty_cup/env_worker.py`:

```python
from RRF_orchestra.workers.env_worker import BaseEnvWorker, BaseEnvWorkerCfg
from RRF_orchestra.protocol.messages  import ObsBatch
from RRF_orchestra.protocol.shared_tensor import SharedTensorRef
from RRF_robotwin_tasks.envs.robotwin_env import RoboTwinEnv
from RoboRenForce.utils.configclass import configclass

class RoboTwinPlaceCupEnvWorker(BaseEnvWorker):
    def setup(self):
        import sapien  # noqa — must happen here, not at module import time
        self._envs = [RoboTwinEnv(task="place_empty_cup", seed=self.cfg.seed + i)
                      for i in range(self.cfg.num_envs_per_worker)]
        self._language = "place the empty cup on the saucer"

    def reset_envs(self, env_ids):
        states, images = [], []
        for i in env_ids:
            obs = self._envs[i].reset()
            states.append(obs["proprio"])
            images.append(obs["images"])
        return ObsBatch(
            worker_ids=[self.cfg.worker_id]*len(env_ids),
            env_ids=env_ids,
            step_ids=[self._step[i] for i in env_ids],
            images=SharedTensorRef.from_tensor(torch.stack(images)),
            states=SharedTensorRef.from_tensor(torch.stack(states)),
            languages=[self._language]*len(env_ids),
            timestamp=time.time(),
        )

    def step_envs(self, action):
        # ... unpack action.actions, step each sub-env, collect rewards/dones
        ...

    def teardown(self):
        for e in self._envs: e.close()

@configclass
class RoboTwinPlaceCupEnvWorkerCfg(BaseEnvWorkerCfg):
    class_type: type[RoboTwinPlaceCupEnvWorker] = RoboTwinPlaceCupEnvWorker
    num_envs_per_worker: int = 4
    seed: int = 0
```

The task package, not the orchestra package, knows what RoboTwin's obs dict looks like. The orchestra package only knows the `ObsBatch` schema.

---

## 5. The orchestrator

`orchestrator/topology.py`:

```python
@configclass
class TopologyCfg(ModuleBaseCfg):
    num_env_workers: int = 4
    env_worker_cfg: BaseEnvWorkerCfg = MISSING       # set by user; e.g. RoboTwinPlaceCupEnvWorkerCfg(...)
    inference_worker_cfg: InferenceWorkerCfg = InferenceWorkerCfg()
    learner_in_process: bool = True                   # v1 default; learner runs in the orchestrator process
    health_timeout_s: float = 10.0
```

`orchestrator/orchestra_runner.py`:

```python
@configclass
class OrchestraVLARunnerCfg(BaseRunnerCfg):
    """Drop-in replacement for VLAGRPORunnerCfg when the user wants orchestra."""
    class_type: type["OrchestraVLARunner"] = "OrchestraVLARunner"
    topology: TopologyCfg = TopologyCfg()
    algorithm: AlgorithmBaseCfg = MISSING            # GRPO/PPO/SAC, same as core
    policy: ActorCriticPackCfg = MISSING

class OrchestraVLARunner(BaseRunner):
    def learn(self):
        with Supervisor(self.cfg.topology) as sup:
            sup.start_all()
            for it in range(self.cfg.max_iterations):
                trajs = sup.collect_trajectories(min_count=self.cfg.batch_size)
                stats = self.algorithm.update(self.policy, trajs)
                if it % self.cfg.weight_sync_every == 0:
                    sup.broadcast_weights(self.policy.state_dict(), version=it)
                self.logger.log_train(it, ..., stats)
            sup.stop_all()
```

The orchestrator IS the learner in v1 (`learner_in_process=True`). Splitting learner into its own process is v2 — same protocol, just different topology.

---

## 6. Step-by-step implementation

### Step 0 — Scaffold the new package (1 h)

- [ ] `source/RRF_orchestra/{pyproject.toml, README.md}` + empty module tree.
- [ ] `pip install -e source/RRF_orchestra` works.
- [ ] `import RRF_orchestra; print(RRF_orchestra.__version__)` works.

### Step 1 — Protocol + transport (3 h)

- [ ] Implement `protocol/shared_tensor.py` — `SharedTensorRef.from_tensor` and `.materialize`. Unit test: round-trip a 224×224×3 uint8 tensor across `mp.Process` boundary, assert no copy (`data_ptr` equality).
- [ ] Implement `protocol/messages.py` — all 5 message types as `@configclass`.
- [ ] Implement `protocol/channels.py` — `Channel.get_batch(max_batch, timeout_s)` is the non-trivial method (drains queue with deadline).
- [ ] Unit test all channels with 2-process round-trips.

### Step 2 — `BaseEnvWorker` + `InferenceWorker` skeletons (3 h)

- [ ] `workers/base_worker.py` — main loop, `should_stop()`, signal handler.
- [ ] `workers/env_worker.py` — abstract methods + provided `run()`.
- [ ] `workers/inference_worker.py` — loads VLA actor, batches obs from `obs_ch.get_batch(...)`, returns actions to per-sim `action_ch[i]`.
- [ ] `workers/batcher.py` — `FixedBatcher(batch_size)` (waits for N obs) + `DynamicBatcher(max_batch, max_wait_ms)`.
- [ ] Standalone test: spawn 1 fake EnvWorker + 1 InferenceWorker with a noop policy; verify obs flows in, actions flow back, both shut down on Ctrl-C.

### Step 3 — Orchestrator + Supervisor (3 h)

- [ ] `orchestrator/topology.py`, `supervisor.py`, `weight_sync.py`.
- [ ] `orchestrator/orchestra_runner.py` — `OrchestraVLARunner.learn()`.
- [ ] Adapter for one core algorithm (`GRPOAlgorithm`) so it can consume `Trajectory` messages.

### Step 4 — First real benchmark: RoboTwin orchestra (3 h)

- [ ] In `RRF_robotwin_vla_rl_tasks/place_empty_cup/`, add `env_worker.py` (concrete `RoboTwinPlaceCupEnvWorker`).
- [ ] Add `agents_grpo_orchestra.py` — `RoboTwinPlaceCupGRPOOrchCfg(OrchestraVLARunnerCfg)` bundling the env worker cfg + policy + GRPO.
- [ ] Register task ID `RoboTwin-PlaceCup-GRPO-Orch-v0`.
- [ ] Run: `python source/RRF_orchestra/RRF_orchestra/examples/train_robotwin_orchestra.py --task RoboTwin-PlaceCup-GRPO-Orch-v0 --num_env_workers 4 --max_iterations 20`
- [ ] **Acceptance**: 20 iters complete without crash; `events.jsonl` shows decreasing policy loss; `inference_worker_batch_size` metric averages > 1 (proves batching works).

### Step 5 — Throughput benchmark (1 h)

- [ ] `examples/benchmark_throughput.py` runs the same task in single-proc and orchestra mode for 60 s and reports steps/s.
- [ ] **Acceptance**: orchestra throughput > 1.5 × single-proc on 1 GPU + 4 CPU sim workers (RoboTwin sapien is CPU-bound).

### Step 6 — Failure handling tests (2 h)

- [ ] Test: kill one EnvWorker mid-training → Supervisor logs "fatal_error from env_worker[2]" and exits cleanly within `health_timeout_s`.
- [ ] Test: InferenceWorker raises in setup → Supervisor exits cleanly (no orphan processes).
- [ ] Test: Ctrl-C → SIGINT propagates, all workers teardown, no zombie processes (`ps -ef | grep python` count returns to baseline).

### Step 7 — Documentation (1 h)

- [ ] `RRF_orchestra/README.md` — "When to use", topology diagram, hello-world example.
- [ ] Update main `README.md` with a section "Orchestra VLA RL training" pointing here.
- [ ] Add a comparison table: single-proc vs orchestra (when to use each).

---

## 7. Acceptance criteria (whole task)

| | Criterion |
|---|---|
| ✅ | `RRF_orchestra` is its own pip-installable package; core repo has zero new files outside it (except per-task `env_worker.py` files, which live in their RRF_<benchmark>_vla_rl packages). |
| ✅ | All cross-process communication goes through `protocol/` channels; no ad-hoc `mp.Queue` use elsewhere. |
| ✅ | Wire protocol is fully documented in `protocol/messages.py` docstrings; `PROTOCOL_VERSION` is checked at worker startup. |
| ✅ | `BaseEnvWorker` is the only customization point; per-task subclasses implement exactly 4 methods (`setup`, `reset_envs`, `step_envs`, `teardown`). |
| ✅ | `RoboTwin-PlaceCup-GRPO-Orch-v0` trains 20 iter end-to-end, throughput ≥ 1.5× single-proc baseline. |
| ✅ | Killing any worker triggers a clean topology shutdown within 10 s, no zombies. |
| ✅ | Default backend is `torch.multiprocessing` (no Ray dependency added to anything). |

---

## 8. Risks & open questions

| Risk / question | Mitigation / decision |
|---|---|
| `mp.Queue` head-of-line blocking when one SimWorker is slow | Per-SimWorker `action_ch[i]`. ObsBatch carries `worker_ids`, InferenceWorker routes accordingly. |
| Shared-mem tensors not freed → OOM | Channels keep weak refs; consumer must call `.materialize()` and the producer's tensor goes out of scope after `put()`. Add a leak test. |
| sapien / mujoco fork-vs-spawn issues | Use `mp.set_start_method("spawn")` in `Supervisor.__init__`. Document this in README. |
| Weight sync stalls inference (large state dict) | First broadcast a `pause` ControlMsg → swap shared-mem handles → broadcast `resume`. Pause window measured in ms. |
| Multi-machine support | v2 only. Channel abstraction makes this swappable: replace `Channel` impl with ZeroMQ/gRPC, no worker-code changes. |
| Weight version drift causing off-policy bias | Algorithm decides via `max_weight_lag` cfg; default `inf` (use everything), but PPO should set this small. |

---

## 9. Open decisions for the user

1. **v1 multi-machine?** I assume **single-machine only** for v1. Confirm.
2. **Learner-in-process?** I assume **yes** for v1 (orchestrator is the learner). The protocol allows splitting later.
3. **Inference worker count = 1?** I assume **yes** for v1. Multi-GPU inference (data-parallel inference workers) is v2 if a single A100 isn't enough.
4. **Weight sync cadence?** Default `every iteration`. PPO needs every iteration; GRPO can lag. Make it per-task config.
5. **Restart-on-crash?** I assume **NO** for v1 (shut down whole topology on any worker death — easier to debug). Add opt-in restart in v2.
