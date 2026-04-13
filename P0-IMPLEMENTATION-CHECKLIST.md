# P0 Implementation Checklist

**MVP Goal**: Working VLA pretraining with 8-GPU DDP support

---

## ✅ Completed: File Skeleton Creation (25 files)

All P0 files created with TODO comments and structure.

---

## 📋 Implementation Roadmap

### Week 0: Data Preparation (Phase 0)

#### File: `scripts/data/isaaclab_to_lerobot.py`
- [ ] Implement Isaac Lab demo loading (HDF5/pickle)
- [ ] Extract observations, actions, rewards
- [ ] Convert images to video (MP4)
- [ ] Write Parquet episode files
- [ ] Compute normalization stats
- [ ] Write stats.safetensors
- [ ] Create info.json metadata
- [ ] **Test**: Convert small test dataset

#### File: `scripts/data/rlds_to_lerobot.py`
- [ ] Implement RLDS dataset loading (TensorFlow datasets)
- [ ] Extract episodes and trajectories
- [ ] Handle different RLDS schemas
- [ ] Convert to LeRobot format
- [ ] **Test**: Convert RLDS sample dataset

---

### Week 1: Data + VLM (Phase 1)

#### File: `dataset/lerobot/lerobot_dataset.py`
- [ ] Load Parquet files
- [ ] Load stats.safetensors
- [ ] Load info.json
- [ ] Implement episode iteration
- [ ] Lazy video decoding (PyAV or OpenCV)
- [ ] Build episode index
- [ ] **Test**: Load and iterate dataset

#### File: `dataset/lerobot/lerobot_processor.py`
- [ ] Implement image transforms (resize, normalize)
- [ ] Implement action normalization
- [ ] Implement proprioception normalization
- [ ] Add text tokenization (optional)
- [ ] **Test**: Process sample batch

#### File: `networks/vlm/vlm_backbone_base.py`
- [ ] Define base VLM interface
- [ ] Add freeze/unfreeze methods
- [ ] **Test**: Subclass works

#### File: `networks/vlm/qwen3vl.py`
- [ ] Load Qwen3-VL from HuggingFace (`transformers`)
- [ ] Setup AutoProcessor
- [ ] Implement feature extraction
- [ ] Add LoRA support (`peft`)
- [ ] Implement freezing
- [ ] **Test**: Load model, extract features

---

### Week 2: Action Expert (Phase 2)

#### File: `networks/vlm/fusion_layers.py`
- [ ] Implement concat + MLP fusion
- [ ] Handle VL + proprioception fusion
- [ ] **Test**: Forward pass with dummy data

#### File: `components/actor/action_heads/diffusion_action_head.py`
- [ ] Implement Transformer noise predictor
- [ ] Register noise schedule (cosine/linear)
- [ ] Implement training forward (add noise + predict)
- [ ] Implement DDIM sampling
- [ ] **Test**: Training forward, DDIM sampling

#### File: `components/actor/action_heads/regression_action_head.py`
- [ ] Implement MLP action head
- [ ] **Test**: Forward pass

#### File: `components/actor/vla_actor.py`
- [ ] Construct VLM backbone from config
- [ ] Construct fusion layer
- [ ] Construct action head
- [ ] Implement forward pass (VLM → fusion → action)
- [ ] Handle frozen VLM
- [ ] **Test**: Full VLA forward pass

---

### Week 3: Single-GPU Training (Phase 3)

#### File: `algorithms/vla_training/pretrain_algorithm.py`
- [ ] Implement action loss (MSE or diffusion loss)
- [ ] Setup gradient scaler (AMP)
- [ ] Implement update step with gradient clipping
- [ ] **Test**: Single update step

#### File: `runners/vla/pretrain/vla_pretrain_runner.py`
- [ ] Load dataset and create DataLoader
- [ ] Construct VLA actor
- [ ] Setup optimizer (AdamW) and scheduler
- [ ] Implement training loop
- [ ] Add validation loop
- [ ] Implement checkpoint saving/loading
- [ ] Integrate TensorBoard logger
- [ ] **Test**: Train for 10 steps

#### File: `RRF_vla_tasks/vla_pretrain/minimal_example.py`
- [ ] Import all config classes
- [ ] Define complete config
- [ ] **Test**: Config validation

#### File: `scripts/vla/pretrain/train_single_gpu.py`
- [ ] Import runner and load_config
- [ ] Implement argument parsing
- [ ] Implement main() function
- [ ] **Test**: Run with minimal_example.py

#### 🎯 Phase 3 Milestone Test
```bash
python scripts/vla/pretrain/train_single_gpu.py \
    --config RRF_vla_tasks.vla_pretrain.minimal_example \
    --num_epochs 1 \
    --batch_size 4
```
**Success Criteria**: Training runs, loss decreases, checkpoint saved

---

### Week 4: DDP Training (Phase 4) ⭐ MVP COMPLETE

#### File: `runners/vla/pretrain/vla_pretrain_runner_distributed.py`
- [ ] Initialize distributed process group
- [ ] Wrap VLA actor with DDP
- [ ] Setup DistributedSampler
- [ ] Override training loop for DDP
- [ ] Add rank 0 logging/saving
- [ ] Implement cleanup
- [ ] **Test**: 2-GPU DDP first

#### File: `RRF_vla_tasks/vla_pretrain/humanoid_qwen3vl.py`
- [ ] Import all config classes
- [ ] Define production config (diffusion head)
- [ ] **Test**: Config validation

#### File: `scripts/vla/pretrain/train_ddp.py`
- [ ] Import distributed runner
- [ ] Implement distributed setup
- [ ] Implement main() function
- [ ] **Test**: Run with 2 GPUs

#### 🎯 Phase 4 Milestone Test (2 GPUs)
```bash
torchrun --nproc_per_node=2 scripts/vla/pretrain/train_ddp.py \
    --config RRF_vla_tasks.vla_pretrain.humanoid_qwen3vl \
    --num_epochs 1 \
    --batch_size 4
```

#### 🎯 Phase 4 Final Test (8 GPUs) ⭐
```bash
torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
    --config RRF_vla_tasks.vla_pretrain.humanoid_qwen3vl \
    --num_epochs 5
```

**Success Criteria**: 
- All 8 GPUs utilized
- Effective batch size = 64 (8 * 8)
- Training stable
- Checkpoints saved
- Loss logged on rank 0

---

## 🧪 Testing Strategy

### Unit Tests (Optional, can be done in parallel)
- Test individual components in isolation
- See: `.claude/project-structure-tests.md`

### Integration Tests (Required for each phase)
- Test complete pipeline for each phase
- Example: Phase 1 → Load dataset, iterate, get sample

### End-to-End Tests (Week 4)
- Full training pipeline
- Data → Model → Training → Checkpoint

---

## 📦 Dependencies

Install before starting implementation:

```bash
pip install torch torchvision
pip install transformers  # Qwen3-VL
pip install peft  # LoRA
pip install pyarrow  # Parquet
pip install safetensors  # Stats
pip install pyav  # Video (or opencv-python)
pip install tensorboard  # Logging
```

---

## 🎯 Success Metrics

### Phase 0-1 Success
- ✅ Convert Isaac Lab demo to LeRobot format
- ✅ Load dataset, iterate, get sample
- ✅ Load Qwen3-VL, extract features

### Phase 2 Success
- ✅ VLA actor forward pass works
- ✅ Action output shape correct

### Phase 3 Success
- ✅ Single-GPU training runs
- ✅ Loss decreases
- ✅ Checkpoint saved and loaded

### Phase 4 Success (MVP Complete) ⭐
- ✅ 8-GPU DDP training stable
- ✅ Effective batch size = 64
- ✅ Training faster than single-GPU (6-7x speedup)
- ✅ Checkpoint compatible with single-GPU

---

## 📞 Help & Resources

**Stuck on implementation?**
1. Check reference code in `.references/`
2. Read detailed specs in `.claude/project-structure-*.md`
3. Check TODO comments in skeleton files

**Architecture questions?**
- See: `.claude/vla-system-architecture.md`
- See: `.claude/vla-training-pipeline.md`

**File locations?**
- See: `.claude/P0-FILES-STATUS.md`

---

## 🎓 Implementation Tips

1. **Start small**: Implement Phase 0 with tiny test dataset
2. **Test early**: Test each component immediately after implementing
3. **Use references**: Copy patterns from `.references/lerobot` and `.references/Psi0`
4. **Debug single-GPU first**: Phase 3 before Phase 4
5. **Check shapes**: Print tensor shapes at each step
6. **Monitor GPU memory**: Use `nvidia-smi` to check utilization

---

**Status**: 📝 Skeleton complete, ready for implementation  
**Next**: Start with `scripts/data/isaaclab_to_lerobot.py` (Phase 0)  
**Goal**: Working 8-GPU VLA pretraining in 4 weeks ⭐
