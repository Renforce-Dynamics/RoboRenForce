# RRF_distributed

Process-level parallelism for VLA RL training. Lives outside `RoboRenForce`
core so single-process users do not pay for `multiprocessing` orchestration.

## Status

- [x] Step 0: package scaffolding
- [x] Step 1: wire protocol (`shared_tensor`, `messages`, `channels`)
- [x] Step 2: `BaseEnvWorker`, `InferenceWorker` skeletons + batchers
- [ ] Step 3: orchestrator + supervisor
- [ ] Step 4: end-to-end RoboTwin distributed run
- [ ] Step 5: throughput benchmark
- [ ] Step 6: failure-handling tests
- [ ] Step 7: user-facing docs

## Install

```bash
pip install -e source/RRF_distributed
```

See `docs/PLAN-task3-distributed-package.md` for the full design.
