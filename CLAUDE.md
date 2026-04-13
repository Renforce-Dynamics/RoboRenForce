# Claude Code Instructions for RoboRenforce

This file contains persistent instructions for Claude Code when working in this repository.

---

## Project Context

RoboRenforce is a robot learning framework combining:
1. **Current state**: Pure RL framework (PPO, SAC, DSAC, GAIL, etc.)
2. **Goal**: Extend to support VLA pretraining + RL fine-tuning

**Key files to review before major changes**:
- `TODO.md` - Task breakdown for VLA integration
- `.claude/project-overview.md` - Architecture and design decisions
- Memory system (auto-loaded) - Reference implementations and patterns

---

## Code Standards

### 1. Always Use Configclass Pattern

All new components MUST follow the `@configclass` pattern:

```python
from RoboRenForce.utils.configclass import configclass, ModuleBaseCfg, MISSING

@configclass
class MyComponentCfg(ModuleBaseCfg):
    class_type: type[MyComponent] = MyComponent
    required_field: int = MISSING  # Forces user to provide
    optional_field: float = 1.0
    nested_cfg: OtherCfg = OtherCfg()  # Auto-wrapped in default_factory
    
class MyComponent(ModuleBase):
    def __init__(self, cfg: MyComponentCfg, *args, **kwargs):
        super().__init__(cfg)
        # Implementation...
```

**Why**: Ensures type safety, serializability, and consistency with existing codebase.

**Validation**: All configs must pass `.validate()` before use. Check for MISSING values.

### 2. Preserve Existing Architecture

**DO NOT** refactor the Runner → Algorithm → Components hierarchy unless explicitly requested.

**DO** extend it:
- New policy types → Add to `ActorCriticPack`
- New algorithms → Subclass `AlgorithmBaseCfg`
- New runners → Subclass `BaseRunnerCfg`

**Example**: VLA actor fits into existing `ActorCriticPackCfg(actor_cfg=VLAActorCfg(...))`

### 3. Reference Implementation Patterns

When implementing from references:

**LeRobot** (`.references/lerobot`):
- **Adopt**: Data format (Parquet + metadata), processor pipeline architecture
- **Adapt**: Use `@configclass` instead of `draccus` for configs
- **Cite**: Add docstring comment referencing source file

**Psi0** (`.references/Psi0`):
- **Adopt**: VLA model architecture (VLM + action head), mixed-dataset training
- **Adapt**: Use `ModuleBaseCfg` instead of Pydantic `BaseModel`
- **Cite**: Reference source file in docstrings

**RoboTwin** (`.references/RoboTwin`):
- **Adopt**: LoRA fine-tuning patterns, environment integration
- **Adapt**: Integrate with RoboRenforce's runner framework
- **Cite**: Reference source file in docstrings

**Template for citing references**:
```python
class VLAActor(ModuleBase):
    """
    VLA actor combining VLM backbone and action head.
    
    Reference: .references/Psi0/src/psi/models/psi0.py
    
    Adaptations:
    - Uses RoboRenforce configclass instead of Pydantic
    - Integrates with ActorCriticPack framework
    """
```

### 4. File Organization

**New VLA components go in**:
- Models: `source/RoboRenForce/networks/vlm/`
- Actors: `source/RoboRenForce/components/actor/vla_actors.py`
- Runners: `source/RoboRenForce/runners/vla_pretrain_runner.py`, `vla_rl_runner.py`
- Data: `source/RoboRenForce/dataset/lerobot_dataset.py`
- Processors: `source/RoboRenForce/utils/processor/`
- Losses: `source/RoboRenForce/algorithms/losses/vla_losses.py`

**Scripts**:
- Training: `scripts/renforce/train_vla_pretrain.py`, `train_vla_rl_finetune.py`
- Data conversion: `scripts/data/rlds_to_lerobot.py`, `isaaclab_to_lerobot.py`

**Examples**:
- Configs: `source/demo_tasks/vla_pretrain/`, `source/demo_tasks/vla_finetune/`

### 5. Testing Requirements

Every new component needs:

1. **Unit test**: Test component in isolation
   ```python
   # tests/components/actor/test_vla_actor.py
   def test_vla_actor_forward():
       cfg = VLAActorCfg(...)
       actor = cfg.construct_from_cfg(dim_params={...})
       output = actor(obs)
       assert output.shape == expected_shape
   ```

2. **Config test**: Serialize/deserialize
   ```python
   def test_vla_actor_cfg_serialization():
       cfg = VLAActorCfg(...)
       cfg_dict = cfg.to_dict()
       cfg_restored = VLAActorCfg()
       cfg_restored.from_dict(cfg_dict)
       assert cfg == cfg_restored
   ```

3. **Integration test**: Test in full pipeline (if applicable)
   ```python
   def test_vla_pretrain_runner():
       runner = VLAPretrainRunner(cfg, env, ...)
       runner.learn(num_iterations=10)  # Short run
       assert runner.logger.has_logged("action_loss")
   ```

**Run tests before committing**: `pytest tests/`

---

## Common Tasks

### Adding a New VLA Model

1. **Create model class** in `source/RoboRenForce/networks/vlm/`:
   ```python
   @configclass
   class MyVLMCfg(ModuleBaseCfg):
       class_type: type[MyVLM] = MyVLM
       model_name: str = "my-vlm-model"
       # ... config fields
   
   class MyVLM(ModuleBase):
       def __init__(self, cfg: MyVLMCfg):
           # Load from HuggingFace, etc.
   ```

2. **Create action head config** in `source/RoboRenForce/components/actor/`:
   ```python
   @configclass
   class MyActionHeadCfg(ModuleBaseCfg):
       # ... config fields
   ```

3. **Create VLA actor config** combining them:
   ```python
   @configclass
   class MyVLAActorCfg(ModuleBaseCfg):
       vlm_cfg: MyVLMCfg = MyVLMCfg()
       action_head_cfg: MyActionHeadCfg = MyActionHeadCfg()
   ```

4. **Write tests** (unit + integration)

5. **Add example config** in `source/demo_tasks/vla_pretrain/my_vla_example.py`

### Adding a New Dataset Format

1. **Create dataset reader** in `source/RoboRenForce/dataset/`:
   ```python
   @configclass
   class MyDatasetCfg(ModuleBaseCfg):
       class_type: type[MyDataset] = MyDataset
       data_root: str = MISSING
       # ... config fields
   
   class MyDataset(torch.utils.data.Dataset):
       def __getitem__(self, idx):
           # Return (obs, action, reward, ...)
   ```

2. **Create format converter** in `scripts/data/`:
   ```python
   # scripts/data/myformat_to_lerobot.py
   def convert_myformat_to_lerobot(input_dir, output_dir):
       # Read myformat data
       # Write to LeRobot Parquet + metadata
   ```

3. **Test converter** on small dataset

4. **Document in TODO.md** and add to Phase 1.3

### Debugging Tips

**Config issues**: 
```python
# Check for MISSING values
missing_fields = cfg.validate()
print(f"Missing: {missing_fields}")

# Inspect serialization
print(cfg.to_dict())
```

**Model loading issues**:
```python
# Check dim_params
print(f"Policy dim: {dim_params['policy_dim']}")
print(f"Action dim: {dim_params['action_dim']}")

# Test forward pass
dummy_obs = torch.randn(batch_size, obs_dim)
output = model(dummy_obs)
print(f"Output shape: {output.shape}")
```

**Runner issues**:
- Check `logger.log_dir` for tensorboard logs
- Add breakpoint in `runner._learn()` to inspect training loop
- Verify environment step output matches expected format

---

## Autonomy Guidelines

### You MAY do automatically:
- Create/modify code files in `source/` following existing patterns
- Write unit tests for new components
- Update TODO.md to check off completed tasks
- Add docstrings and type hints
- Refactor within a single component (no cross-component changes)

### You MUST ask before:
- Changing the configclass system (`utils/configclass/`)
- Modifying base classes (`ModuleBase`, `BaseRunner`, `AlgorithmBase`)
- Adding new dependencies to requirements.txt
- Changing training script interfaces (`scripts/renforce/train_*.py`)
- Large refactoring across multiple files (>5 files)

### When stuck:
1. Check memory files for relevant patterns
2. Read reference implementation source code
3. Ask user with specific question: "I'm implementing X, should I follow pattern A (from LeRobot) or pattern B (from Psi0)?"

---

## Git Workflow

**Before committing**:
1. Run tests: `pytest tests/`
2. Check imports: Ensure no circular dependencies
3. Format code: Follow existing style (4 spaces, PEP8)

**Commit messages**:
- Use conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Reference TODO items: `feat(vla): implement LeRobot dataset reader (#TODO Phase 1.1)`

**Branch naming**:
- Feature branches: `feature/vla-pretrain`, `feature/lerobot-dataset`
- Bugfix branches: `fix/config-serialization`

---

## Performance Considerations

### Memory Optimization
- Use gradient checkpointing for large VLA models
- Lazy load datasets (don't load all into RAM)
- Clear cache after validation: `torch.cuda.empty_cache()`

### Training Speed
- Use mixed precision (fp16/bf16) for VLA training
- Batch mixture sampling (don't iterate datasets sequentially)
- Profile with `torch.profiler` if slow

### Disk I/O
- Parquet files for efficient storage
- Video codec: libsvtav1 (best compression) or h264 (fast)
- Lazy video decoding (only load frames when needed)

---

## Checklist for VLA Integration Tasks

Before marking a TODO item as complete:

- [ ] Code follows configclass pattern
- [ ] Unit tests written and passing
- [ ] Config serialization tested (`.to_dict()` / `.from_dict()`)
- [ ] Docstrings include reference to source (if adapted)
- [ ] Integrated with existing runners (if applicable)
- [ ] Example config added (if user-facing)
- [ ] TODO.md updated

---

## Questions?

**First check**:
1. Memory files (`.claude/projects/.../memory/`)
2. `.claude/project-overview.md`
3. TODO.md
4. Reference implementations (`.references/`)

**Still stuck?** Ask user with context:
- What you're trying to do
- What you've tried
- Specific decision point or blocker