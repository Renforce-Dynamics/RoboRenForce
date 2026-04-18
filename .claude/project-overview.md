# RoboRenforce Project Overview

**Purpose**: Unified framework for robot learning combining VLA pretraining and RL fine-tuning

---

## Current State (April 2025)

RoboRenforce is a **pure RL framework** for robotics with:
- On-policy algorithms: PPO, MBPO, SAPG, EPO
- Off-policy algorithms: SAC, DSAC (distributional variants)
- Imitation learning: GAIL, AMP, distillation
- Isaac Lab / Gymnasium environment integration
- Modular configclass-based architecture

**No VLA support yet** - this is what we're adding.

---

## Vision: VLA + RL Integration

### Three-Stage Pipeline

```
[Stage 1: VLA Pretraining]
  ↓ LeRobot dataset (Parquet) + Mixed datasets
  ↓ VLM backbone (Qwen3-VL) + Action head (Diffusion/Regression)
  ↓ Pretrained VLA checkpoint
  
[Stage 2: RL Environment Setup]
  ↓ Isaac Lab tasks with vision + language
  ↓ Reward function definition
  
[Stage 3: RL Fine-tuning]
  ↓ Load pretrained VLA
  ↓ LoRA fine-tuning with PPO/SAC
  ↓ Task-specific policy
```

### Key Features (Planned)

1. **Data Infrastructure**
   - LeRobot Parquet format support
   - Processor pipeline for image/text/state transforms
   - Mixed-dataset training (combine multiple sources)
   - Format converters (RLDS, Isaac Lab → LeRobot)

2. **VLA Models**
   - VLM backbones: Qwen3-VL, OpenVLA, custom
   - Action heads: Regression (L1), Diffusion (DDIM)
   - Action tokenization for discrete binning
   - LoRA support for parameter-efficient training

3. **Pretraining**
   - VLAPretrainRunner with mixed-dataset sampling
   - Action prediction + language modeling losses
   - Checkpoint management with HuggingFace Hub

4. **RL Fine-tuning**
   - VLA as drop-in actor in existing runners
   - LoRA fine-tuning (freeze VLM, train action head)
   - Exploration strategies for VLA policies
   - Task-specific reward modeling

---

## Architecture Principles

### Configclass System

All components use `@configclass` decorator for configuration:
```python
@configclass
class VLAActorCfg(ModuleBaseCfg):
    class_type: type[VLAActor] = VLAActor
    vlm_backbone_cfg: VLMBackboneCfg = VLMBackboneCfg(...)
    action_head_cfg: ActionHeadCfg = MISSING
    
actor = VLAActorCfg(...).construct_from_cfg(dim_params=...)
```

**Benefits**: Type-safe, serializable, composable, validated

### Runner → Algorithm → Components

```
Runner (training loop orchestration)
  ├─ Environment (Isaac Lab / Gym)
  ├─ Policy (VLA or traditional actor-critic)
  ├─ Algorithm (PPO, SAC, or VLA pretrain)
  ├─ Replay Buffer / Rollout Storage
  └─ Logger
```

**VLA fits in as**: New policy type in ActorCriticPack, new runner for pretraining

---

## Reference Implementations

### LeRobot (`.references/lerobot`)
**Use for**: Data format, dataset loading, processor pipeline
- Parquet-based episode storage
- Lazy video decoding
- Processor registry architecture
- HuggingFace Hub integration

**Key files**:
- `lerobot/datasets/lerobot_dataset.py` - Dataset class
- `lerobot/processor/pipeline.py` - Data transforms
- `lerobot/configs/train.py` - Training configs

### Psi0 (`.references/Psi0`)
**Use for**: VLA model architecture, pretraining pipeline
- Qwen3-VL backbone + Diffusion Transformer action head
- Mixed-dataset training (EgoDex + real robot data)
- Action tokenization and chunking

**Key files**:
- `psi/models/psi0.py` - VLA model
- `psi/trainers/pretrain.py` - Pretrain trainer
- `psi/config/data_mix.py` - Mixed dataset configs

### RoboTwin (`.references/RoboTwin`)
**Use for**: LoRA fine-tuning, RLDS format, task benchmarks
- OpenVLA fine-tuning with LoRA
- RLDS dataset format
- Environment integration patterns

**Key files**:
- `policy/openvla-oft/vla-scripts/finetune.py` - Fine-tuning script
- `envs/` - Environment wrappers

---

## Development Roadmap

See [TODO.md](../TODO.md) for detailed task breakdown.

**High-level phases**:
1. **Phase 1-2**: Data infrastructure + VLA models (foundational)
2. **Phase 3**: VLA pretraining pipeline (core capability)
3. **Phase 4**: RL fine-tuning integration (unique value-add)
4. **Phase 5-6**: Testing + documentation (production-ready)

**Critical path**: LeRobot dataset loader → VLA actor implementation → Pretrain runner → RL fine-tuning

---

## Technical Decisions

### Data Format: LeRobot Parquet
**Rationale**: 
- Episode-aware structure
- Efficient storage with lazy loading
- Standard normalization stats format
- Wide adoption in VLA community

**Alternative considered**: RLDS (used by RoboTwin)
- Rejected: Less efficient, harder to extend

### Config System: Stick with @configclass
**Rationale**:
- Already proven in RoboRenforce
- Type-safe, serializable, validated
- Avoids mixing multiple config frameworks

**Alternative considered**: Adopt `draccus` (LeRobot) or `pydantic` (Psi0)
- Rejected: Would require refactoring all existing code

### VLA Integration: ActorCriticPack Extension
**Rationale**:
- Minimal changes to existing runners
- VLA becomes just another actor type
- Reuse all existing RL algorithms

**Alternative considered**: Separate VLA-specific runners
- Rejected: Code duplication, harder to maintain

### LoRA for Fine-tuning
**Rationale**:
- Parameter-efficient (train <5% of params)
- Preserves pretrained knowledge
- Faster convergence than full fine-tuning

**Alternative considered**: Full fine-tuning
- Rejected: Risk of catastrophic forgetting, slower

---

## Success Metrics

### Phase 3 (Pretraining)
- [ ] Action MSE < 0.1 on validation set (comparable to Psi0)
- [ ] Training throughput > 100 samples/sec on single GPU
- [ ] Can load/save checkpoints compatible with HF Hub

### Phase 4 (RL Fine-tuning)
- [ ] Task success rate improves by >20% vs. pretrained VLA alone
- [ ] Fine-tuning converges in <10k env steps
- [ ] LoRA updates preserve pretrained knowledge (validation MSE stable)

### Overall
- [ ] End-to-end pipeline: raw data → pretrained VLA → fine-tuned policy
- [ ] Reproducible configs for all stages
- [ ] Documentation enables external users to replicate

---

## Known Risks & Mitigations

**Risk 1**: LeRobot data format incompatible with our tasks
- **Mitigation**: Create converters early (Phase 1.3), validate on small datasets

**Risk 2**: VLA models too large for RL fine-tuning
- **Mitigation**: Use LoRA + freeze VLM backbone, optimize memory with gradient checkpointing

**Risk 3**: RL fine-tuning degrades pretrained performance
- **Mitigation**: Monitor validation metrics, use small LoRA rank, add KL penalty to RL loss

**Risk 4**: Config system complexity explodes
- **Mitigation**: Follow existing patterns strictly, validate configs early in validate()

---

## Collaboration Guidelines

### When Adding New Components

1. **Follow configclass pattern**:
   - Create `XxxCfg(ModuleBaseCfg)` with `class_type` field
   - Implement `construct_from_cfg()` if custom logic needed
   - Use `MISSING` for required fields

2. **Add to registry** (if applicable):
   - Processor steps → `@register_processor_step`
   - Algorithms → Add to `AlgorithmBaseCfg` subclasses
   - Runners → Add to `BaseRunnerCfg` subclasses

3. **Write tests**:
   - Unit test: Component forward pass
   - Integration test: Component in full pipeline
   - Config test: Serialize/deserialize via `.to_dict()` / `.from_dict()`

### Code Review Checklist

- [ ] Follows existing configclass patterns?
- [ ] Type annotations complete (or auto-inferred)?
- [ ] Uses `construct_from_cfg()` for instantiation?
- [ ] Config validated with `.validate()`?
- [ ] References added to docstrings (e.g., "Reference: .references/lerobot/...")?
- [ ] Works with both Isaac Lab and Gymnasium environments?

---

## Contact & Resources

**Primary working directory**: `/home/ununtu/code/RoboRenforce`

**Additional directories** (for reference):
- `/home/ununtu/code/hvla/.reference/Psi0`
- `/home/ununtu/code/hvla/.reference/RoboTwin`
- (LeRobot at `.references/lerobot` in main repo)

**Documentation**:
- [TODO.md](../TODO.md) - Task breakdown
- This file - High-level overview
- Memory system - `.claude/projects/.../memory/` (auto-loaded context)

**Questions?** Check memory files first:
- `project_roborenforce_architecture.md` - Current RL framework
- `project_configclass_design.md` - Config system details
- `reference_lerobot_integration.md` - LeRobot data format
- `reference_psi0_vla_model.md` - VLA model architecture
- `reference_robotwin_rl_finetuning.md` - RL fine-tuning patterns