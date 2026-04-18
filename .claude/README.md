# RoboRenForce VLA Integration Documentation

This directory contains comprehensive documentation for VLA (Vision-Language-Action) integration into the RoboRenForce framework.

---

## 📚 Documentation Index

### Architecture & Design

1. **[Project Overview](project-overview.md)**
   - High-level vision for VLA + RL integration
   - Design principles and goals
   - Reference to external implementations

2. **[VLA System Architecture](vla-system-architecture.md)**
   - Three-system architecture (System 2: VLM, System 1: Action Expert, System 0: Locomotion)
   - Distributed training design (DDP)
   - Training paradigm differences (VLA data parallelism vs RL environment parallelism)

3. **[VLA Training Pipeline](vla-training-pipeline.md)**
   - Three-stage training: Pretrain → Post-Train → RL Fine-tune
   - Hyperparameter guidelines
   - Monitoring metrics

### Implementation Structure

4. **[Core Framework Structure](project-structure-core.md)** ⭐
   - Root: `source/RoboRenForce/RoboRenForce/`
   - Detailed breakdown of:
     - Algorithms (on-policy, off-policy, VLA training)
     - Buffers (rollout storage, replay buffers)
     - Components (actors, critics, VLA actors, action heads)
     - Dataset (LeRobot format, processors)
     - Networks (MLP, Transformer, VLM backbones)
     - Runners (RL runners, VLA runners)
     - Utils (configclass, env wrappers, VLA utilities)

5. **[VLA Tasks Structure](project-structure-vla-tasks.md)**
   - Root: `source/RRF_vla_tasks/RRF_vla_tasks/`
   - Example configs for:
     - VLA pretraining (minimal, full production, baselines)
     - Post-training (SFT, DPO)
     - RL fine-tuning (PPO, SAC)

6. **[Scripts Structure](project-structure-scripts.md)** ⭐
   - Root: `scripts/`
   - Complete script organization:
     - `scripts/renforce/` - Low-level RL (unchanged)
     - `scripts/vla/` - VLA training (21 files)
       - pretrain/ (single-GPU, DDP, DeepSpeed)
       - post_train/ (SFT, DPO)
       - rl/ (PPO/SAC fine-tuning)
       - eval/ (evaluation scripts)
       - utils/ (visualization, export)
     - `scripts/distributed/` - DDP launchers (4 files)
     - `scripts/data/` - Data conversion (6 files)

7. **[Tests Structure](project-structure-tests.md)**
   - Root: `tests/`
   - Comprehensive test organization:
     - Unit tests (components, networks, datasets, algorithms)
     - Integration tests (runners, end-to-end)
     - Testing strategy and CI/CD

### Implementation Roadmap

8. **[Implementation Roadmap](implementation-roadmap.md)** ⭐
   - 7-phase implementation plan (Weeks 0-7)
   - MVP definition (Phases 0-4, 27 files)
   - Priority breakdown (P0/P1/P2)
   - Risk mitigation strategies

9. **[Scripts Organization Design](scripts-organization.md)**
   - Rationale for VLA/RL separation
   - Usage examples for each training stage
   - Distributed training commands

---

## 🎯 Quick Navigation by Phase

### Phase 0: Preparation (Week 0)
- Read: [Implementation Roadmap](implementation-roadmap.md) - Section "Phase 0"
- Files: Data conversion scripts in [Scripts Structure](project-structure-scripts.md) - Section 4

### Phase 1: Data + VLM Loading (Week 1)
- Read: [Core Framework Structure](project-structure-core.md) - Sections 4 (Dataset) & 5 (Networks/VLM)
- Files: 8 files (4 P0, 4 P1)

### Phase 2: Action Expert (Week 2)
- Read: [Core Framework Structure](project-structure-core.md) - Section 3.2 (VLA Components)
- Files: 5 files (3 P0, 2 P1)

### Phase 3: Single-GPU Training (Week 3)
- Read: [Scripts Structure](project-structure-scripts.md) - Section 2.1 (Pretrain Scripts)
- Read: [VLA Tasks Structure](project-structure-vla-tasks.md) - Section 1 (Pretrain Configs)
- Files: 6 files

### Phase 4: DDP Training (Week 4) ⭐ **CRITICAL**
- Read: [VLA System Architecture](vla-system-architecture.md) - DDP section
- Read: [Scripts Structure](project-structure-scripts.md) - Sections 2.1 & 3 (DDP + Distributed)
- Files: 8 files

### Phase 5: Post-Train (Week 5)
- Read: [VLA Training Pipeline](vla-training-pipeline.md) - SFT section
- Read: [Scripts Structure](project-structure-scripts.md) - Section 2.2 (Post-Train)
- Files: 4 files

### Phase 6: RL Fine-tuning (Week 6)
- Read: [VLA Training Pipeline](vla-training-pipeline.md) - RL Fine-tune section
- Read: [Scripts Structure](project-structure-scripts.md) - Section 2.3 (RL Scripts)
- Files: 4 files

### Phase 7: System 0 Interface (Week 7+)
- Read: [VLA System Architecture](vla-system-architecture.md) - System 0 section
- Files: 2 files

---

## 📊 File Count Summary

| Component | Total Files | P0 (Critical) | P1 (Important) | P2 (Future) |
|-----------|-------------|---------------|----------------|-------------|
| **Core Framework** | ~35 | 15 | 12 | 8 |
| **VLA Tasks** | 11 | 3 | 4 | 4 |
| **Scripts** | 31 | 5 | 12 | 14 |
| **Tests** | 25 | 5 | 6 | 14 |
| **Total** | **~102** | **28** | **34** | **40** |

**MVP (P0 only)**: 28 files (Phases 0-4)

---

## 🔧 Usage Examples

### Pretrain VLA (Single-GPU)
```bash
python scripts/vla/pretrain/train_single_gpu.py \
    --config RRF_vla_tasks.vla_pretrain.minimal_example \
    --log_dir logs/pretrain_single
```

### Pretrain VLA (8 GPUs, DDP)
```bash
torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
    --config RRF_vla_tasks.vla_pretrain.humanoid_qwen3vl \
    --log_dir logs/pretrain_ddp
```

### SFT Post-Training (4 GPUs)
```bash
torchrun --nproc_per_node=4 scripts/vla/post_train/train_sft.py \
    --config RRF_vla_tasks.vla_post_train.sft_humanoid_reach \
    --pretrained checkpoints/humanoid_qwen3vl/vla_step_10000.pth \
    --log_dir logs/sft
```

### RL Fine-tuning (PPO, 4096 envs)
```bash
python scripts/vla/rl/train_ppo_finetune.py \
    --config RRF_vla_tasks.vla_rl_finetune.ppo_humanoid_reach \
    --pretrained checkpoints/sft_reach/vla_step_5000.pth \
    --task Isaac-Humanoid-Reach-v0 \
    --num_envs 4096 \
    --log_dir logs/rl_finetune
```

---

## 🚀 Getting Started

1. **First Time**: Read [Project Overview](project-overview.md) and [Implementation Roadmap](implementation-roadmap.md)

2. **Understanding Architecture**: Read [VLA System Architecture](vla-system-architecture.md)

3. **Implementation**: Follow [Implementation Roadmap](implementation-roadmap.md) phase-by-phase
   - Start with Phase 0 (Data Prep)
   - Then Phase 1 (Data + VLM)
   - Continue through Phase 4 (MVP)

4. **Reference**: Use structure documents as lookup
   - Code organization: [Core Framework Structure](project-structure-core.md)
   - Scripts: [Scripts Structure](project-structure-scripts.md)
   - Tests: [Tests Structure](project-structure-tests.md)

---

## 📝 Key Design Principles

1. **Separation of Concerns**
   - VLA code in separate directories from low-level RL
   - `scripts/vla/` vs `scripts/renforce/`
   - `RRF_vla_tasks/` vs `RRF_isaaclab_tasks/`

2. **ConfigClass Pattern**
   - All components use `@configclass` decorator
   - Type-safe, serializable configs
   - `construct_from_cfg()` for instantiation

3. **Three-System Architecture**
   - System 2: VLM backbone (frozen, reusable)
   - System 1: Action Expert (trainable, core target)
   - System 0: Locomotion (existing RL)

4. **Progressive Training**
   - Pretrain (offline data) → Post-Train (SFT) → RL Fine-tune
   - Each stage builds on previous

5. **Distributed Training**
   - VLA: Data parallelism (DDP, 8-16 GPUs)
   - RL: Environment parallelism (4096 envs, 1-4 GPUs)
   - Clear separation of distributed strategies

---

## 🔄 Document Update Log

- **2026-04-13**: Reorganized into focused documents
  - Split monolithic `project-structure.md` into:
    - `project-structure-core.md` (Core framework)
    - `project-structure-vla-tasks.md` (Task configs)
    - `project-structure-scripts.md` (Scripts)
    - `project-structure-tests.md` (Tests)
  - Aligned with renamed directory structure (`RRF_vla_tasks`)
  - Updated file counts and priorities

- **2026-04-13**: Initial documentation created
  - Project overview, VLA architecture, training pipeline
  - Implementation roadmap, scripts organization

---

## 📞 Questions?

If you're implementing a specific phase:
1. Check the phase section in [Implementation Roadmap](implementation-roadmap.md)
2. Read the relevant structure document (core/tasks/scripts/tests)
3. Reference [VLA System Architecture](vla-system-architecture.md) for design decisions

For architecture questions:
- Three-system design → [VLA System Architecture](vla-system-architecture.md)
- Training stages → [VLA Training Pipeline](vla-training-pipeline.md)
- Distributed training → [VLA System Architecture](vla-system-architecture.md) DDP section

For implementation details:
- Code structure → [Core Framework Structure](project-structure-core.md)
- Script usage → [Scripts Structure](project-structure-scripts.md)
- Testing → [Tests Structure](project-structure-tests.md)
