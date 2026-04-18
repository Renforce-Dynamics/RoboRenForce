# VLA Implementation Roadmap - Executive Summary

基于三层System架构 + 分布式训练的精简实施方案

---

## 🎯 核心架构调整

### 旧设计 → 新设计对比

| 维度 | 旧设计 | 新设计 | 理由 |
|-----|--------|--------|------|
| **VLM训练** | 从头预训练 | ✅ 复用Qwen3-VL | 省资源，VLM已在ego数据上预训练 |
| **核心目标** | VLM + Action Head都训练 | ✅ 只训练Action Expert (System 1) | 机器人特定，必须训练 |
| **训练范式** | 单卡环境并行 | ✅ 多卡DDP (8-16 GPUs) | VLA需要数据并行，不是环境并行 |
| **System层级** | 两层 (VLA + RL) | ✅ 三层 (System 2/1/0) | 参考Psi0，更清晰 |
| **文件数量** | ~120 files | ✅ ~70 files | 精简，专注核心 |

---

## 🏗️ 三层System架构

```
┌────────────────────────────────────────────────────┐
│           Three-System VLA Architecture             │
└────────────────────────────────────────────────────┘

System 2: VLM (Vision-Language Feature Extractor)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Qwen3-VL-2B-Instruct (from HuggingFace)
🔧 Strategy: ✅ 复用预训练 ❌ 不从头训练
💾 Status: 冻结或LoRA微调
📊 Data: 已在EgoDex预训练（~100k trajectories）

            ↓ VL Embeddings

System 1: Action Expert (Robot-Specific Action Head)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Diffusion Transformer / Regression MLP
🔧 Strategy: ⭐ 核心训练目标！多卡DDP (8-16 GPUs)
💾 Training: Pretrain → Post-Train → RL (三阶段)
📊 Data: 机器人数据 (HE_RAW, RoboTwin, custom)

            ↓ High-level Actions

System 0: Locomotion Controller (Low-Level Tracking)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 PPO-trained locomotion policy
🔧 Strategy: ✅ 假设已有 (RoboRenforce现有功能)
💾 Status: 不需要重新训练，只需接口
📊 Data: -
```

---

## 📊 训练阶段对比

| 阶段 | 训练内容 | 分布式策略 | GPU需求 | 优先级 | 周期 |
|------|---------|-----------|---------|--------|------|
| System 2 Pretrain | VLM backbone | DDP/FSDP | 32-64 | ❌ 跳过 | - |
| **System 1 Pretrain** | Action Expert | **DDP** | **8-16** | **⭐⭐⭐ P0** | **Week 1-4** |
| System 1 Post-Train | SFT/DPO | DDP | 4-8 | ⭐⭐ P1 | Week 5 |
| System 1 RL | PPO/SAC | Env并行 | 1-4 | ⭐ P2 | Week 6 |
| System 0 Interface | Tracking | - | - | ⭐ P3 | Week 7 |

---

## 🚀 7周实施计划

### Week 0: 准备 (Phase 0)
```
□ 调研预训练VLM (Qwen3-VL vs OpenVLA)
□ 搭建8-GPU训练环境
□ 测试DDP基础功能
□ 阅读Psi0参考代码

Deliverable: 环境就绪
```

### Week 1-2: 数据 + VLM加载 (Phase 1) ⭐⭐⭐
```
Files: 8个

□ dataset/lerobot/lerobot_dataset.py      # LeRobot数据加载
□ dataset/lerobot/metadata.py
□ utils/processor/pipeline.py              # 数据处理管道
□ networks/vlm/qwen3_vl.py                 # 加载预训练VLM ⭐
□ networks/vlm/vlm_loader.py               # 统一VLM加载器

Deliverable: 
  ✓ 可以加载LeRobot数据
  ✓ 可以加载Qwen3-VL (冻结)
  ✓ VLM forward pass工作
```

### Week 2-3: Action Expert (Phase 2) ⭐⭐⭐
```
Files: 5个

□ networks/action_heads/action_expert.py   # System 1核心 ⭐
□ networks/action_heads/diffusion_head.py  # Diffusion Transformer
□ networks/action_heads/regression_head.py # MLP baseline
□ components/actor/vla_actors.py           # 三层组合

Deliverable:
  ✓ Action Expert forward pass工作
  ✓ VLA Actor (System 2 + System 1) 工作
```

### Week 3: 单卡训练验证 (Phase 3) ⭐⭐⭐
```
Files: 6个

□ algorithms/vla_pretrain/vla_pretrain_base.py
□ algorithms/losses/vla_losses.py
□ runners/vla/pretrain/pretrain_runner.py  # 单卡版本
□ scripts/renforce/train_vla_pretrain.py
□ demo_tasks/vla_pretrain/minimal_example.py

Deliverable:
  ✓ 单卡训练System 1成功
  ✓ Action loss下降
  ✓ 架构验证完成

Command:
  python scripts/renforce/train_vla_pretrain.py \
      --config demo_tasks/vla_pretrain/minimal_example.py
```

### Week 4: 多卡分布式 (Phase 4) ⭐⭐⭐ 核心里程碑
```
Files: 8个

□ utils/distributed/ddp_utils.py           # DDP工具
□ utils/distributed/checkpoint_utils.py
□ runners/vla/pretrain/distributed_pretrain_runner.py  # ⭐ 核心
□ scripts/renforce/train_vla_pretrain_distributed.py
□ scripts/distributed/launch_ddp.py

Deliverable:
  ✓ 8卡DDP训练System 1
  ✓ 训练速度提升8x
  ✓ Gradient sync正常
  ✓ 生产级训练流程

Command:
  torchrun --nproc_per_node=8 \
      scripts/renforce/train_vla_pretrain_distributed.py \
      --config demo_tasks/vla_pretrain/humanoid_qwen3vl.py
```

### Week 5: Post-Train (Phase 5) ⭐⭐
```
Files: 4个

□ runners/vla/post_train/sft_runner.py
□ runners/vla/post_train/distributed_sft_runner.py
□ scripts/renforce/train_vla_post_train.py

Deliverable:
  ✓ SFT on task-specific demos
  ✓ Task success rate提升

Command:
  torchrun --nproc_per_node=4 \
      scripts/renforce/train_vla_post_train.py \
      --pretrained checkpoints/pretrained_system1.pth
```

### Week 6: RL Fine-tune (Phase 6) ⭐
```
Files: 4个

□ algorithms/vla_pretrain/vla_rl_finetune.py
□ runners/vla/rl/rl_on_policy_runner.py
□ utils/env_wrapper/vla_wrapper/multimodal_env_wrapper.py
□ scripts/renforce/train_vla_rl_finetune.py

Deliverable:
  ✓ RL fine-tune VLA
  ✓ 环境并行 (4096 envs)
  ✓ Reward最大化

Command:
  python scripts/renforce/train_vla_rl_finetune.py \
      --task Isaac-Humanoid-Reach-v0 \
      --pretrained checkpoints/sft_system1.pth
```

### Week 7: System 0 (Phase 7) ⭐
```
Files: 2个

□ components/locomotion/locomotion_tracker.py
□ utils/env_wrapper/vla_wrapper/hierarchical_wrapper.py

Deliverable:
  ✓ System 1 → System 0 tracking
  ✓ 全身控制
```

---

## 📦 关键文件清单

### MVP (Minimum Viable Product) - Week 0-4
**27个核心文件** = 可以多卡训练System 1

```
Phase 1: Data + VLM (8 files)
├── dataset/lerobot/lerobot_dataset.py
├── dataset/lerobot/metadata.py
├── dataset/lerobot/stats_utils.py
├── utils/processor/processor_base.py
├── utils/processor/pipeline.py
├── networks/vlm/vlm_base.py
├── networks/vlm/qwen3_vl.py
└── networks/vlm/vlm_loader.py

Phase 2: Action Expert (5 files)
├── networks/action_heads/action_expert.py     ⭐
├── networks/action_heads/diffusion_head.py
├── networks/action_heads/regression_head.py
├── components/actor/vla_actors.py
└── components/normalizer/image_normalizer.py

Phase 3: Single-GPU Training (6 files)
├── algorithms/vla_pretrain/vla_pretrain_base.py
├── algorithms/losses/vla_losses.py
├── runners/vla/pretrain/pretrain_runner.py
├── scripts/renforce/train_vla_pretrain.py
├── demo_tasks/vla_pretrain/minimal_example.py
└── tests/test_runners/test_vla_pretrain_runner.py

Phase 4: Distributed Training (8 files) ⭐
├── utils/distributed/ddp_utils.py
├── utils/distributed/checkpoint_utils.py
├── utils/distributed/logging_utils.py
├── utils/distributed/launch_utils.py
├── runners/vla/pretrain/distributed_pretrain_runner.py  ⭐⭐⭐
├── scripts/renforce/train_vla_pretrain_distributed.py
├── scripts/distributed/launch_ddp.py
└── configs/distributed/ddp_config.yaml
```

### Full Pipeline - Week 0-7
**37个核心文件** + 测试 + 文档 = **~70 files**

---

## 💡 关键技术决策

### 决策1: System 2复用 ❌不训练 ✅冻结
**为什么**: 
- Qwen3-VL已经在大规模ego数据预训练
- 从头训练需要32-64 GPUs × 数周
- 冻结VLM只需要训练Action Expert (~500M params)

**如何实现**:
```python
vlm = Qwen3VLBackbone(freeze=True)  # 默认冻结
# 或
vlm = Qwen3VLBackbone(freeze=False, lora_rank=8)  # LoRA微调
```

### 决策2: System 1是核心 ⭐ 必须多卡DDP
**为什么**:
- 机器人特定，必须训练
- 数据量: 1k-10k trajectories (可控)
- 模型大小: ~500M params (Diffusion Transformer)
- 单卡太慢，多卡DDP必需

**如何实现**:
```bash
# 8卡DDP训练
torchrun --nproc_per_node=8 train_vla_pretrain_distributed.py
```

### 决策3: RL保持环境并行 ✅单卡learner
**为什么**:
- RL阶段模型固定，主要瓶颈是环境模拟
- 4096并行envs比数据并行更高效
- Isaac Lab环境并行已优化

**如何实现**:
```python
# 单卡learner + 4096 parallel envs
runner = VLARLOnPolicyRunner(num_envs=4096)
```

### 决策4: DDP优先，DeepSpeed备选
**为什么**:
- DDP简单，PyTorch原生
- System 1不大 (~500M)
- 如果不冻结VLM (2B)，再用DeepSpeed ZeRO-3

**如何实现**:
```python
# Phase 4: DDP
model = DDP(model, device_ids=[local_rank])

# Phase 8 (optional): DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(model=model)
```

### 决策5: 渐进实现 ✅单卡验证 → 多卡DDP
**为什么**:
- 先单卡验证架构可行（Phase 3）
- 再上多卡DDP加速（Phase 4）
- 避免一上来就调试分布式问题

**实施顺序**:
```
Week 3: 单卡训练 (验证loss下降)
   ↓
Week 4: 多卡DDP (加速到8x)
```

---

## 🔥 核心里程碑

### ✅ Milestone 1: Data + VLM Ready (Week 2)
```
能够:
- 加载LeRobot数据
- 加载Qwen3-VL (冻结)
- VLM forward pass

验证命令:
  python -c "from dataset.lerobot import LeRobotDataset; \
             from networks.vlm import Qwen3VLBackbone; \
             print('OK')"
```

### ✅ Milestone 2: VLA Actor Works (Week 3)
```
能够:
- VLA Actor (System 2 + 1) forward
- 输出action predictions

验证命令:
  python -c "from components.actor import VLAActor; \
             actor = ...; \
             actions = actor(obs_dict); \
             print(actions.shape)"
```

### ✅ Milestone 3: Single-GPU Training (Week 3)
```
能够:
- 单卡训练System 1
- Action loss下降
- 保存checkpoint

验证命令:
  python scripts/renforce/train_vla_pretrain.py \
      --config demo_tasks/vla_pretrain/minimal_example.py
  # 观察loss下降
```

### 🎯 Milestone 4: Multi-GPU DDP Training (Week 4) ⭐⭐⭐
```
能够:
- 8卡DDP训练System 1
- Gradient sync正常
- 速度提升8x
- 生产级训练流程

验证命令:
  torchrun --nproc_per_node=8 \
      scripts/renforce/train_vla_pretrain_distributed.py \
      --config demo_tasks/vla_pretrain/humanoid_qwen3vl.py
  
  # 验证:
  # 1. 8个进程启动
  # 2. Loss同步
  # 3. Throughput提升
```

**这是最关键的里程碑！达成后即可大规模训练System 1**

---

## 📈 预期效果

### 训练性能

| 配置 | Throughput | 训练时间 (10k steps) | 成本 |
|------|-----------|---------------------|------|
| 单卡 (A100) | ~50 samples/sec | 55 hours | 高 |
| 8卡 DDP (A100) | ~400 samples/sec | 7 hours | ⭐ 推荐 |
| 16卡 DDP (A100) | ~800 samples/sec | 3.5 hours | 可选 |

### 模型性能

| 阶段 | Validation Action MSE | Task Success Rate |
|------|----------------------|-------------------|
| 预训练后 (Pretrain) | <0.1 | 60-70% |
| 后训练后 (SFT) | <0.05 | 80-90% |
| RL微调后 | <0.05 | 90-95%+ |

---

## ⚠️ 风险与缓解

### 风险1: DDP调试困难
**缓解**: 
- Week 3先单卡验证
- Week 4再上DDP
- 充分测试sync机制

### 风险2: VLM加载失败
**缓解**:
- 提前测试HuggingFace下载
- 准备离线权重
- 多个VLM备选方案

### 风险3: GPU资源不足
**缓解**:
- MVP只需单卡（Week 3可完成）
- 8卡DDP可选（加速用）
- 支持gradient accumulation模拟大batch

### 风险4: 数据格式不兼容
**缓解**:
- Week 1优先验证数据加载
- 准备格式转换脚本
- 参考LeRobot实现

---

## 🎯 下一步行动

### 立即开始 (本周)

1. **环境准备** (Day 1-2)
   ```bash
   # 安装依赖
   pip install torch transformers peft safetensors pyarrow
   
   # 测试DDP
   torchrun --nproc_per_node=2 test_ddp.py
   ```

2. **数据准备** (Day 3-5)
   ```bash
   # 下载示例数据
   # 或转换现有数据到LeRobot格式
   python scripts/data/rlds_to_lerobot.py --input ... --output ...
   ```

3. **VLM测试** (Day 6-7)
   ```python
   # 测试加载Qwen3-VL
   from transformers import Qwen3VLForConditionalGeneration
   model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
   ```

### Week 1 目标

- [ ] 完成Phase 1前4个文件
- [ ] LeRobot dataset可以加载
- [ ] Qwen3-VL可以forward

**这周的重点是打通数据流和VLM加载！**

---

## 📚 参考资源

- **Psi0**: `.references/Psi0/` - 三层system设计
- **LeRobot**: `.references/lerobot/` - 数据格式和训练
- **PyTorch DDP**: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- **NCCL**: https://docs.nvidia.com/deeplearning/nccl/

---

**总结**: 这个方案更务实，专注核心（System 1训练），复用已有（System 2 VLM），渐进实现（单卡→多卡），文件精简（70个核心文件），7周完成。
