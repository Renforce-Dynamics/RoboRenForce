# RoboRenforce VLA Integration TODO (Updated)

**架构更新**: 基于三层System架构 + 分布式训练需求

**核心理念**:
- System 2 (VLM): 复用预训练模型（Qwen3-VL）
- System 1 (Action Expert): **核心训练目标**，多卡DDP
- System 0 (Locomotion): 假设已有，只需接口

**训练范式**:
- Pretrain/Post-Train: 多卡DDP（8-16 GPUs）
- RL Fine-tune: 环境并行（4096 envs）+ 单卡learner

---

## Phase 0: 准备工作 (Week 0)

### 0.1 环境调研
- [ ] **调研预训练VLM选择**
  - [ ] Qwen3-VL-2B-Instruct (Alibaba)
  - [ ] OpenVLA-7B (Stanford)
  - [ ] PaliGemma-3B (Google)
  - 决策标准: 模型大小、性能、社区支持
  
- [ ] **搭建多卡训练环境**
  - [ ] 准备8+ GPU机器（训练System 1）
  - [ ] 安装PyTorch 2.0+ with NCCL
  - [ ] 测试DDP基础功能
  - [ ] 配置共享存储（NFS/GPFS）

- [ ] **阅读参考代码**
  - [ ] `.references/Psi0/src/psi/models/psi0.py` - 三层system设计
  - [ ] `.references/Psi0/src/psi/trainers/pretrain.py` - 预训练流程
  - [ ] `.references/lerobot` - 数据格式和分布式训练

**Outcome**: 环境就绪，理解架构 ✓

---

## Phase 1: 数据基础 + VLM加载 (Week 1-2) ⭐⭐⭐

### 1.1 LeRobot Dataset Loader
- [ ] **Create dataset module**
  - [ ] `source/RoboRenForce/RoboRenForce/dataset/lerobot/__init__.py`
  - [ ] `dataset/lerobot/lerobot_dataset.py` - 主dataset类
    - `LeRobotDatasetCfg(ModuleBaseCfg)`
    - `LeRobotDataset(torch.utils.data.Dataset)`
    - 支持Parquet加载 + episode索引
    - 支持delta_timestamps配置
  - [ ] `dataset/lerobot/metadata.py` - 元数据解析
    - `LeRobotMetadata` class
    - 解析`dataset_metadata.json`
  - [ ] `dataset/lerobot/stats_utils.py` - 统计加载
    - `load_stats(stats_path)` - 加载safetensors
    - `compute_stats(dataset)` - 计算normalization stats
  - Reference: `.references/lerobot/src/lerobot/datasets/lerobot_dataset.py`

- [ ] **Create processor pipeline**
  - [ ] `utils/processor/__init__.py`
  - [ ] `utils/processor/processor_base.py` - ProcessorStep基类
    - `ProcessorStepCfg(ModuleBaseCfg)`
    - `ProcessorStep(ModuleBase)`
    - Processor registry装饰器
  - [ ] `utils/processor/pipeline.py` - 管道组合
    - `DataProcessorPipelineCfg`
    - `DataProcessorPipeline` - 链式调用processors
  - [ ] `utils/processor/image_processors.py` - 图像处理
    - `ResizeImageStep`
    - `NormalizeImageStep`
    - `ToTensorStep`
  - Reference: `.references/lerobot/src/lerobot/processor/`

- [ ] **Test data loading**
  - [ ] `tests/test_dataset/test_lerobot_dataset.py`
    - Test Parquet loading
    - Test metadata parsing
    - Test processor pipeline

**Outcome**: 可以加载LeRobot数据 ✓

### 1.2 VLM Backbone Loading (System 2)
- [ ] **Create VLM loading infrastructure**
  - [ ] `networks/vlm/__init__.py`
  - [ ] `networks/vlm/vlm_base.py` - VLM抽象接口
    - `VLMBackboneCfg(ModuleBaseCfg)`
    - `VLMBackbone(ModuleBase)`
    - 定义接口: `forward(images, text) -> embeddings`
  
  - [ ] `networks/vlm/qwen3_vl.py` - Qwen3-VL实现 ⭐
    - `Qwen3VLBackboneCfg(VLMBackboneCfg)`
      - `model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct"`
      - `freeze: bool = True`  # 默认冻结
      - `lora_rank: int = 0`   # 0=冻结, >0=LoRA
    - `Qwen3VLBackbone(VLMBackbone)`
      - 从HuggingFace加载模型
      - 可选冻结参数
      - 可选LoRA微调
  
  - [ ] `networks/vlm/vlm_loader.py` - 统一加载器 ⭐
    - `load_pretrained_vlm(cfg: VLMBackboneCfg) -> VLMBackbone`
    - 处理不同VLM的加载逻辑
    - 自动下载权重（如果需要）
  
  - Reference: `.references/Psi0/src/psi/models/psi0.py` (Qwen3-VL使用)

- [ ] **Test VLM loading**
  - [ ] `tests/test_networks/test_vlm_backbone.py`
    - Test Qwen3-VL加载
    - Test forward pass
    - Test冻结/LoRA切换

**Outcome**: 可以加载预训练Qwen3-VL，forward pass工作 ✓

---

## Phase 2: Action Expert (System 1) (Week 2-3) ⭐⭐⭐

### 2.1 Action Expert Core
- [ ] **Create Action Expert module** ⭐ 核心！
  - [ ] `networks/action_heads/__init__.py`
  
  - [ ] `networks/action_heads/action_expert.py` - System 1主类
    - `ActionExpertCfg(ModuleBaseCfg)`
      - `architecture: str = "diffusion"`  # "diffusion", "regression"
      - `action_chunk_size: int = 16`
      - `action_dim: int = MISSING`
      - `robot_type: str = "humanoid"`
    - `ActionExpert(ModuleBase)`
      - 根据architecture选择head
      - `forward(embeddings) -> actions [B, chunk_size, action_dim]`
      - `sample(embeddings, num_steps) -> actions` (DDIM采样)
  
  - [ ] `networks/action_heads/diffusion_head.py` - Diffusion Transformer
    - `DiffusionHeadCfg(ModuleBaseCfg)`
      - `num_diffusion_steps: int = 100`
      - `dit_hidden_dim: int = 512`
      - `dit_num_layers: int = 6`
    - `DiffusionTransformerHead(ModuleBase)`
      - Time embeddings
      - DiT blocks with AdaLayerNorm
      - DDIM sampler
    - Reference: `.references/Psi0/src/psi/models/psi0.py` (Diffusion部分)
  
  - [ ] `networks/action_heads/regression_head.py` - 简单MLP
    - `RegressionHeadCfg(ModuleBaseCfg)`
    - `RegressionHead(ModuleBase)`
      - MLP: embeddings -> actions
  
  - [ ] `networks/action_heads/action_head_base.py` - 基类
    - `ActionHeadCfg(ModuleBaseCfg)`
    - `ActionHead(ModuleBase)`

- [ ] **Test Action Expert**
  - [ ] `tests/test_networks/test_action_expert.py`
    - Test diffusion head forward
    - Test DDIM sampling
    - Test regression head

**Outcome**: Action Expert forward/sample工作 ✓

### 2.2 VLA Actor (三层组合)
- [ ] **Create VLA Actor** ⭐
  - [ ] `components/actor/vla_actors.py` - 组合三层system
    - `VLAActorCfg(ModuleBaseCfg)`
      - `vlm_backbone_cfg: VLMBackboneCfg = MISSING`  # System 2
      - `action_expert_cfg: ActionExpertCfg = MISSING` # System 1
      - `freeze_vlm: bool = True`
      - `use_proprioception: bool = True`
    - `VLAActor(ModuleBase)`
      - 加载VLM (System 2, 冻结)
      - 创建Action Expert (System 1)
      - Fusion layer: VLM embedding + proprioception
      - `forward(obs_dict) -> actions`
        - obs_dict包含: image, text, proprioception

- [ ] **Create image normalizer**
  - [ ] `components/normalizer/image_normalizer.py`
    - `ImageNormalizerCfg(NormalizerBaseCfg)`
    - `ImageNormalizer(NormalizerBase)`
    - ImageNet mean/std

- [ ] **Test VLA Actor**
  - [ ] `tests/test_components/test_vla_actor.py`
    - Test三层forward pass
    - Test VLM冻结
    - Test action输出shape

**Outcome**: VLA Actor (System 2+1) forward pass工作 ✓

---

## Phase 3: 单卡训练验证 (Week 3) ⭐⭐⭐

### 3.1 Pretrain Algorithm
- [ ] **Create pretrain algorithm**
  - [ ] `algorithms/vla_pretrain/__init__.py`
  - [ ] `algorithms/vla_pretrain/vla_pretrain_base.py`
    - `VLAPretrainAlgorithmCfg(AlgorithmBaseCfg)`
      - `learning_rate: float = 3e-5`
      - `warmup_steps: int = 1000`
      - `max_grad_norm: float = 1.0`
    - `VLAPretrainAlgorithm(AlgorithmBase)`
      - `update(batch) -> update_info_dict`
      - Optimizer + Scheduler setup
  
  - [ ] `algorithms/losses/vla_losses.py`
    - `VLALossCfg(ModuleBaseCfg)`
    - `VLALoss(nn.Module)`
      - Action prediction loss (L1/L2/Diffusion)
      - `forward(predictions, targets) -> loss_dict`

### 3.2 Single-GPU Pretrain Runner
- [ ] **Create pretrain runner** (单卡版本)
  - [ ] `runners/vla/__init__.py`
  - [ ] `runners/vla/pretrain/__init__.py`
  - [ ] `runners/vla/pretrain/pretrain_runner.py`
    - `VLAPretrainRunnerCfg(BaseRunnerCfg)`
      - `vla_actor_cfg: VLAActorCfg = MISSING`
      - `dataset_cfg: LeRobotDatasetCfg = MISSING`
      - `batch_size: int = 32`
      - `num_epochs: int = 100`
      - `validation_freq: int = 5`
      - `save_freq: int = 10`
    - `VLAPretrainRunner(BaseRunner)`
      - `learn(num_epochs)` - 训练循环
      - `validate()` - 验证
      - `save_checkpoint()` - 保存
    - Reference: `.references/Psi0/src/psi/trainers/pretrain.py`

### 3.3 Training Script
- [ ] **Create training entry point**
  - [ ] `scripts/renforce/train_vla_pretrain.py`
    - 单卡训练入口
    - Parse args
    - Create runner
    - Run training

- [ ] **Create minimal config**
  - [ ] `source/demo_tasks/vla_pretrain/__init__.py`
  - [ ] `source/demo_tasks/vla_pretrain/minimal_example.py`
    - `MinimalVLAPretrainCfg(VLAPretrainRunnerCfg)`
    - Qwen3-VL + Regression head
    - 小数据集测试

### 3.4 Test
- [ ] **Test pretrain runner**
  - [ ] `tests/test_runners/test_vla_pretrain_runner.py`
    - Test training loop
    - Test checkpoint save/load
    - Test validation

**Outcome**: 单卡训练System 1成功，loss下降 ✓

---

## Phase 4: 分布式训练 (Week 4) ⭐⭐⭐

### 4.1 DDP Infrastructure
- [ ] **Create distributed utils**
  - [ ] `utils/distributed/__init__.py`
  
  - [ ] `utils/distributed/ddp_utils.py` - DDP helper
    - `init_distributed(backend, rank, world_size)`
    - `cleanup_distributed()`
    - `is_main_process()`
    - `barrier()`
  
  - [ ] `utils/distributed/checkpoint_utils.py` - 分布式checkpoint
    - `save_checkpoint_ddp(model, optimizer, path, rank)`
    - `load_checkpoint_ddp(model, optimizer, path)`
    - 只在rank 0保存/加载
  
  - [ ] `utils/distributed/logging_utils.py` - 分布式logging
    - `DistributedLogger` - 只在rank 0 log
  
  - [ ] `utils/distributed/launch_utils.py` - Launcher
    - `setup_torch_distributed()` - 从环境变量读取
    - `get_world_info()` - 获取rank, world_size等

- [ ] **Create mixed precision utils**
  - [ ] `utils/mixed_precision/__init__.py`
  - [ ] `utils/mixed_precision/amp_utils.py`
    - `AMPContext` - autocast wrapper
    - `GradScaler` wrapper

### 4.2 Distributed Pretrain Runner
- [ ] **Create distributed runner** ⭐ 核心！
  - [ ] `runners/vla/pretrain/distributed_pretrain_runner.py`
    - `DistributedVLAPretrainRunnerCfg(VLAPretrainRunnerCfg)`
      - `backend: str = "nccl"`
      - `world_size: int = MISSING`
      - `rank: int = MISSING`
      - `local_rank: int = MISSING`
      - `use_amp: bool = True`
      - `amp_dtype: str = "bfloat16"`
      - `gradient_accumulation_steps: int = 1`
    - `DistributedVLAPretrainRunner(VLAPretrainRunner)`
      - `_init_distributed()` - 初始化DDP
      - Wrap model with DDP
      - DistributedSampler for dataloader
      - Mixed precision training
      - `learn()` - 分布式训练循环
        - Gradient accumulation
        - Sync across GPUs
        - Only log/save on rank 0

### 4.3 Distributed Training Scripts
- [ ] **Create distributed training entry**
  - [ ] `scripts/renforce/train_vla_pretrain_distributed.py`
    - DDP训练入口
    - 从环境变量读取rank/world_size
    - 创建DistributedVLAPretrainRunner
  
  - [ ] `scripts/distributed/__init__.py`
  - [ ] `scripts/distributed/launch_ddp.py` - DDP launcher
    - `torchrun` wrapper
    - 设置环境变量
    - 启动多进程
  
  - [ ] `scripts/distributed/launch_slurm.py` - SLURM launcher (optional)
    - SLURM集群支持

### 4.4 Config Files
- [ ] **Create distributed configs**
  - [ ] `configs/distributed/ddp_config.yaml`
    - DDP默认配置
  - [ ] `configs/distributed/slurm_template.sh`
    - SLURM脚本模板

### 4.5 Test
- [ ] **Test DDP training**
  - [ ] `tests/test_distributed/test_ddp_utils.py`
    - Test distributed setup
  - [ ] Integration test: 2卡DDP训练10 steps

**Outcome**: 8卡DDP训练System 1，速度提升8x ✓

---

## Phase 5: Post-Train (Week 5) ⭐⭐

### 5.1 SFT Runner
- [ ] **Create post-train infrastructure**
  - [ ] `runners/vla/post_train/__init__.py`
  
  - [ ] `runners/vla/post_train/post_train_runner.py` - 基类
    - `VLAPostTrainRunnerCfg(BaseRunnerCfg)`
      - `pretrained_vla_path: str = MISSING`
      - `freeze_vlm: bool = True`
      - `lora_rank: int = 8`
    - `VLAPostTrainRunner(BaseRunner)`
      - 加载预训练VLA
      - 应用LoRA
  
  - [ ] `runners/vla/post_train/sft_runner.py` - SFT
    - `VLASFTRunnerCfg(VLAPostTrainRunnerCfg)`
      - `regularization_weight: float = 0.1`  # KL penalty
    - `VLASFTRunner(VLAPostTrainRunner)`
      - Task-specific demonstrations
      - KL to pretrained
  
  - [ ] `runners/vla/post_train/distributed_sft_runner.py` - 多卡SFT
    - DDP版本的SFT

### 5.2 SFT Script
- [ ] **Create SFT training script**
  - [ ] `scripts/renforce/train_vla_post_train.py`
    - SFT训练入口

### 5.3 Example Config
- [ ] **Create SFT example**
  - [ ] `source/demo_tasks/vla_post_train/__init__.py`
  - [ ] `source/demo_tasks/vla_post_train/humanoid_sft_example.py`

**Outcome**: SFT on task data, task success提升 ✓

---

## Phase 6: RL Fine-tune (Week 6) ⭐

### 6.1 RL Extensions
- [ ] **Create RL fine-tune algorithm**
  - [ ] `algorithms/vla_pretrain/vla_rl_finetune.py`
    - `VLARLAlgorithmCfg` - 扩展PPOCfg
      - `kl_penalty_coef: float = 0.01`
      - `pretrain_aux_loss_coef: float = 0.1`
    - RL loss + KL penalty

- [ ] **Create RL actor**
  - [ ] Update `components/actor/vla_actors.py`
    - `VLARLActorCfg(VLAActorCfg)`
      - `pretrained_vla_path: str = MISSING`
      - `freeze_vlm: bool = True`
      - `rl_lora_rank: int = 8`
      - `exploration_noise_std: float = 0.1`
    - `VLARLActor(VLAActor)`
      - Load pretrained VLA
      - Add LoRA to action head
      - Exploration noise
      - KL to pretrained

### 6.2 RL Runner
- [ ] **Create VLA RL runner**
  - [ ] `runners/vla/rl/__init__.py`
  - [ ] `runners/vla/rl/rl_on_policy_runner.py`
    - `VLARLOnPolicyRunnerCfg(OnPolicyRunnerCfg)`
      - `pretrained_vla_path: str = MISSING`
      - `action_chunk_size: int = 1`
      - `kl_penalty_coef: float = 0.01`
    - `VLARLOnPolicyRunner(OnPolicyRunner)`
      - 扩展OnPolicyRunner
      - Load pretrained VLA
      - Action chunking
      - Receding horizon control

### 6.3 Multimodal Env Wrapper
- [ ] **Create VLA env wrapper**
  - [ ] `utils/env_wrapper/vla_wrapper/__init__.py`
  - [ ] `utils/env_wrapper/vla_wrapper/multimodal_env_wrapper.py`
    - `MultimodalEnvWrapper`
      - Wrap Isaac Lab env
      - Render images from cameras
      - Provide language instructions
      - Bundle proprioception
      - `get_observations() -> {image, text, proprioception}`

### 6.4 RL Script
- [ ] **Create RL training script**
  - [ ] `scripts/renforce/train_vla_rl_finetune.py`
    - RL fine-tune入口
    - 环境并行（4096 envs）

### 6.5 Example Config
- [ ] **Create RL example**
  - [ ] `source/demo_tasks/vla_finetune/__init__.py`
  - [ ] `source/demo_tasks/vla_finetune/humanoid_ppo_finetune.py`

**Outcome**: RL fine-tune VLA, task reward最大化 ✓

---

## Phase 7: System 0 Interface (Week 7) ⭐

### 7.1 Locomotion Interface
- [ ] **Create locomotion tracker**
  - [ ] `components/locomotion/__init__.py`
  - [ ] `components/locomotion/locomotion_tracker.py`
    - `LocomotionTrackerCfg(ModuleBaseCfg)`
    - `LocomotionTracker(ModuleBase)`
      - Load locomotion policy (PPO trained)
      - Track high-level actions
      - Output joint torques

- [ ] **Update VLA Actor for System 0**
  - [ ] Update `components/actor/vla_actors.py`
    - Add `use_locomotion_tracking: bool`
    - Add `locomotion_policy_path: str`
    - If enabled, System 1 actions → System 0 tracking

### 7.2 Hierarchical Wrapper
- [ ] **Create hierarchical wrapper**
  - [ ] `utils/env_wrapper/vla_wrapper/hierarchical_wrapper.py`
    - Wrap env for hierarchical control
    - High-level actions (System 1) → Low-level (System 0)

**Outcome**: Full three-system pipeline ✓

---

## Phase 8: Advanced Features (Week 8+) ⭐ (Optional)

### 8.1 DeepSpeed Support (If needed)
- [ ] `runners/vla/pretrain/deepspeed_pretrain_runner.py`
- [ ] `configs/distributed/deepspeed_config.json`
- Use case: 如果需要不冻结VLM (2B params)

### 8.2 DPO Post-Train
- [ ] `runners/vla/post_train/dpo_runner.py`
- [ ] Preference dataset format
- Use case: Human preference alignment

### 8.3 More VLM Backbones
- [ ] `networks/vlm/openvla.py` - OpenVLA-7B
- [ ] `networks/vlm/paligemma.py` - PaliGemma-3B

### 8.4 Production Tools
- [ ] `scripts/model/export_to_onnx.py` - ONNX export
- [ ] `scripts/model/benchmark_inference.py` - 推理benchmark
- [ ] `scripts/model/quantization.py` - INT8 quantization

---

## Testing & Documentation

### Tests (Ongoing)
- [ ] Unit tests for all modules (~20 test files)
- [ ] Integration test: End-to-end pretrain
- [ ] Integration test: End-to-end RL fine-tune
- [ ] Distributed test: 2-GPU DDP

### Documentation
- [ ] `docs/tutorials/01_data_preparation.md`
- [ ] `docs/tutorials/02_vla_pretrain.md`
- [ ] `docs/tutorials/03_distributed_training.md`
- [ ] `docs/tutorials/04_sft_post_train.md`
- [ ] `docs/tutorials/05_rl_finetune.md`
- [ ] `docs/api/` - API documentation
- [ ] Example notebooks

---

## Summary: Critical Path

```
Phase 0: 环境准备 (Week 0)
    ↓
Phase 1: 数据 + VLM加载 (Week 1-2) ← 8 files
    ↓
Phase 2: Action Expert (Week 2-3) ← 5 files
    ↓
Phase 3: 单卡训练验证 (Week 3) ← 6 files
    ↓
Phase 4: 分布式训练 (Week 4) ← 8 files ⭐ 核心里程碑
    ↓
Phase 5: Post-Train (Week 5) ← 4 files
    ↓
Phase 6: RL Fine-tune (Week 6) ← 4 files
    ↓
Phase 7: System 0 (Week 7) ← 2 files
```

**Minimum Viable Product (MVP)**: Phase 0-4 (4 weeks, 27 files)
- 可以多卡训练System 1
- Action Expert工作

**Full Pipeline**: Phase 0-7 (7 weeks, 37 files)
- Pretrain → Post-Train → RL
- Three-system integration

**Total Files**: ~37 core files + ~20 tests + ~10 docs = **~70 files** (精简版)

---

## Key Design Decisions

✅ **System 2 (VLM)**: 复用预训练，冻结或LoRA
✅ **System 1 (Action Expert)**: 核心训练目标，多卡DDP
✅ **System 0 (Locomotion)**: 假设已有，低优先级接口
✅ **DDP优先**: Phase 4核心，DeepSpeed备选
✅ **渐进实现**: 单卡验证 → 多卡DDP → RL
✅ **精简文件**: 从120个减少到~70个核心文件

