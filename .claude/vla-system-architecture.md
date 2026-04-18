# VLA System Architecture: Three-Level Design + Distributed Training

**核心Insight**:
1. VLA = System 2 (VLM) + System 1 (Action Expert) + System 0 (Locomotion)
2. System 2可复用，System 1需训练，System 0假设已有
3. 训练需要多卡DDP/FSDP支持（与RL单卡环境并行不同）

---

## 🏗️ Three-System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       VLA SYSTEM HIERARCHY                       │
└─────────────────────────────────────────────────────────────────┘

System 2: VLM Backbone (Vision-Language Feature Extractor)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Model: Qwen3-VL-2B-Instruct / OpenVLA / SigLIP+T5
📊 Pretrained on: Large-scale ego-centric data (EgoDex ~100k trajectories)
🎯 Role: Extract multimodal embeddings from (image, text) → embedding
🔧 Training: 
   - Option A: Use off-the-shelf (Qwen3-VL, OpenVLA) ← RECOMMENDED
   - Option B: Pretrain from scratch (需要大规模GPU集群)
💾 Output: vision_language_embedding [B, seq_len, hidden_dim]

策略: ✅ 直接复用 Hugging Face pretrained models
      ✅ 冻结或LoRA fine-tune
      ❌ 不从头训练（资源消耗太大）

         ↓ Embeddings

System 1: Action Expert (Robot-Specific Action Head)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Model: Diffusion Transformer / Regression MLP
📊 Data: Robot-specific demonstrations (LeRobot format)
   - Humanoid manipulation: HE_RAW (~1k trajectories)
   - Bimanual tasks: RoboTwin datasets
   - Custom collected data
🎯 Role: embeddings + proprioception → action_predictions
🔧 Training Pipeline:
   Stage 1 - Pretrain: Train on mixed robot datasets (多卡DDP)
   Stage 2 - Post-Train: Fine-tune on task-specific demos (多卡DDP)
   Stage 3 - RL: Fine-tune with reward signal (多卡环境并行 + 单卡learner)
💾 Output: actions [B, action_chunk_size, action_dim]

策略: ⭐ 核心训练目标！需要分布式训练支持
      ✅ Pretrain on diverse robot data
      ✅ Post-train on task data
      ✅ RL fine-tune (optional)

         ↓ High-level Actions

System 0: Locomotion Controller (Low-Level Tracking)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Model: RL-trained locomotion policy (PPO/AMP)
📊 Trained on: Locomotion tasks (walking, balancing)
🎯 Role: Track high-level actions with whole-body control
🔧 Training: Already done ← 假设已有
💾 Output: joint_torques [num_joints]

策略: ✅ 假设已有（RoboRenforce现有RL功能）
      ✅ 不需要重新训练
      ✅ 只需要接口对接
```

---

## 🎯 Training Paradigm Shift

### ❌ 旧范式: RL Locomotion Training (现有RoboRenforce)

```
Single GPU/Node
    ↓
Vectorized Environments (4096 parallel envs)
    ↓
On-Policy RL (PPO)
    ↓
Fast iteration (1M steps/hour)

特点:
- 环境并行为主（4096 Isaac Lab envs）
- 单卡或少量卡（1-4 GPUs）
- On-policy算法（PPO, TRPO）
- 计算瓶颈: 环境模拟
```

### ✅ 新范式: VLA System 1 Training (需要实现)

```
Multiple GPUs (8-64 GPUs)
    ↓
Distributed Data Parallel (DDP/FSDP)
    ↓
Large-scale Dataset (100k+ trajectories)
    ↓
Supervised/Self-supervised Learning
    ↓
Slower iteration (depends on dataset size)

特点:
- 数据并行为主（DDP across GPUs）
- 多卡训练必需（8+ GPUs for 2B VLM）
- 监督学习为主（Pretrain, Post-train）
- 计算瓶颈: 模型forward/backward
```

---

## 📊 训练阶段对比表

| 阶段 | 训练内容 | 数据类型 | 分布式策略 | GPU需求 | 优先级 |
|------|---------|---------|-----------|---------|--------|
| **System 2 Pretrain** | VLM backbone | Ego数据 (100k+) | DDP/FSDP | 32-64 GPUs | ❌ 跳过（复用） |
| **System 1 Pretrain** | Action Expert | 机器人数据 (1k-10k) | DDP | 8-16 GPUs | ⭐⭐⭐ P0 |
| **System 1 Post-Train** | Task adaptation | 任务演示 (100-1k) | DDP | 4-8 GPUs | ⭐⭐ P1 |
| **System 1 RL** | Reward optimize | 在线交互 | Env并行 + 单卡 | 1-4 GPUs | ⭐ P2 |
| **System 0** | Locomotion | - | - | - | ✅ 已有 |

---

## 🔧 架构设计调整

### 1. VLA Actor 三层结构

```python
@configclass
class VLAActorCfg(ModuleBaseCfg):
    """三层VLA actor配置"""
    # System 2: VLM Backbone
    vlm_backbone_cfg: VLMBackboneCfg = MISSING
    freeze_vlm: bool = True              # 默认冻结（复用预训练）
    vlm_lora_rank: int = 0               # 0=冻结, >0=LoRA微调
    
    # System 1: Action Expert
    action_expert_cfg: ActionExpertCfg = MISSING
    action_chunk_size: int = 16          # 预测N步轨迹
    
    # System 0: Locomotion (接口)
    use_locomotion_tracking: bool = False
    locomotion_policy_path: Optional[str] = None
    
    # Fusion
    fusion_cfg: FusionLayerCfg = FusionLayerCfg()
    use_proprioception: bool = True


class VLAActor(ModuleBase):
    """三层VLA actor"""
    def __init__(self, cfg: VLAActorCfg, dim_params):
        # System 2: Load pretrained VLM (冻结或LoRA)
        self.vlm = self._load_pretrained_vlm(cfg.vlm_backbone_cfg)
        if cfg.freeze_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif cfg.vlm_lora_rank > 0:
            self.vlm = self._apply_lora(self.vlm, cfg.vlm_lora_rank)
        
        # System 1: Action expert (需要训练)
        self.action_expert = cfg.action_expert_cfg.construct_from_cfg(...)
        
        # System 0: Locomotion policy (可选)
        if cfg.use_locomotion_tracking:
            self.locomotion_policy = self._load_locomotion_policy(
                cfg.locomotion_policy_path
            )
        
        # Fusion
        self.fusion_layer = cfg.fusion_cfg.construct_from_cfg(...)
    
    def forward(self, obs_dict):
        """
        obs_dict = {
            "image": [B, C, H, W],
            "text": [B] list of str,
            "proprioception": [B, proprio_dim],  # 机器人状态
        }
        """
        # System 2: VLM embedding
        with torch.no_grad() if self.cfg.freeze_vlm else nullcontext():
            vl_embedding = self.vlm(
                images=obs_dict["image"],
                text=obs_dict["text"]
            )  # [B, seq_len, hidden_dim]
        
        # Fusion: VLM + proprioception
        fused = self.fusion_layer(
            vl_embedding, 
            obs_dict["proprioception"]
        )  # [B, fusion_dim]
        
        # System 1: Action expert
        high_level_actions = self.action_expert(fused)  # [B, chunk_size, action_dim]
        
        # System 0: Locomotion tracking (if enabled)
        if self.cfg.use_locomotion_tracking:
            low_level_actions = self.locomotion_policy.track(
                high_level_actions[:, 0, :]  # 当前步
            )
            return low_level_actions
        else:
            return high_level_actions[:, 0, :]  # 返回第一步动作
```

### 2. Action Expert 专门设计

```python
@configclass
class ActionExpertCfg(ModuleBaseCfg):
    """System 1: Action Expert配置"""
    class_type: type = ActionExpert
    
    # 架构选择
    architecture: str = "diffusion"  # "diffusion", "regression", "tokenized"
    
    # Diffusion Transformer (推荐)
    dit_hidden_dim: int = 512
    dit_num_layers: int = 6
    dit_num_heads: int = 8
    num_diffusion_steps: int = 100
    
    # Action chunking
    action_chunk_size: int = 16
    action_dim: int = MISSING
    
    # 机器人特定
    robot_type: str = "humanoid"  # "humanoid", "bimanual", "quadruped"


class ActionExpert(ModuleBase):
    """
    System 1: 机器人特定的动作专家
    
    这是VLA的核心训练目标！
    """
    def __init__(self, cfg: ActionExpertCfg):
        if cfg.architecture == "diffusion":
            self.head = DiffusionTransformerHead(cfg)
        elif cfg.architecture == "regression":
            self.head = RegressionHead(cfg)
        else:
            raise ValueError(f"Unknown architecture: {cfg.architecture}")
    
    def forward(self, embeddings):
        """
        embeddings: [B, embedding_dim] from fusion layer
        返回: actions [B, chunk_size, action_dim]
        """
        return self.head(embeddings)
    
    def sample(self, embeddings, num_inference_steps=20):
        """DDIM sampling for diffusion"""
        if isinstance(self.head, DiffusionTransformerHead):
            return self.head.sample(embeddings, num_inference_steps)
        else:
            return self.forward(embeddings)
```

---

## 🚀 分布式训练支持

### 核心需求

```
1. Data Parallel (DDP): 多卡训练System 1
2. Model Parallel (FSDP): 处理大模型（如果VLM不冻结）
3. Gradient Accumulation: 支持大batch size
4. Mixed Precision (FP16/BF16): 加速训练
5. Checkpointing: 断点续训
```

### 实现方案

#### 方案A: PyTorch DDP (推荐Phase 1)

```python
# runners/vla/pretrain/distributed_pretrain_runner.py

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

@configclass
class DistributedVLAPretrainRunnerCfg(VLAPretrainRunnerCfg):
    """分布式预训练配置"""
    # 分布式设置
    backend: str = "nccl"  # "nccl", "gloo"
    world_size: int = MISSING  # 总GPU数
    rank: int = MISSING  # 当前进程rank
    local_rank: int = MISSING  # 本地GPU id
    
    # DDP设置
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    
    # 混合精度
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16", "bfloat16"
    
    # 梯度累积
    gradient_accumulation_steps: int = 1
    
    # 同步设置
    sync_batch_norm: bool = False


class DistributedVLAPretrainRunner(VLAPretrainRunner):
    """
    分布式VLA预训练Runner
    支持多卡DDP训练
    """
    def __init__(self, cfg: DistributedVLAPretrainRunnerCfg, log_dir, device):
        # 初始化分布式环境
        self._init_distributed(cfg)
        
        # 调用父类初始化
        super().__init__(cfg, log_dir, device)
        
        # Wrap model with DDP
        self.vla_actor = DDP(
            self.vla_actor,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            find_unused_parameters=cfg.find_unused_parameters,
        )
        
        # Mixed precision scaler
        if cfg.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _init_distributed(self, cfg):
        """初始化分布式环境"""
        dist.init_process_group(
            backend=cfg.backend,
            init_method='env://',  # 从环境变量读取
            world_size=cfg.world_size,
            rank=cfg.rank,
        )
        torch.cuda.set_device(cfg.local_rank)
        
        # 只在rank 0打印
        self.is_main_process = (cfg.rank == 0)
    
    def _create_dataloader(self):
        """创建分布式dataloader"""
        dataset = self.cfg.dataset_cfg.construct_from_cfg()
        
        # 分布式sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.cfg.world_size,
            rank=self.cfg.rank,
            shuffle=True,
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        
        return dataloader, sampler
    
    def learn(self, num_epochs):
        """分布式训练循环"""
        for epoch in range(num_epochs):
            # 设置epoch（影响sampler的shuffle）
            self.sampler.set_epoch(epoch)
            
            # 训练一个epoch
            for batch_idx, batch in enumerate(self.dataloader):
                # Mixed precision forward
                with torch.cuda.amp.autocast(
                    enabled=self.cfg.use_amp,
                    dtype=getattr(torch, self.cfg.amp_dtype)
                ):
                    loss_dict = self._compute_loss(batch)
                
                # Backward with gradient scaling
                loss = loss_dict["total_loss"]
                if self.cfg.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
                    if self.cfg.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Log (only rank 0)
                if self.is_main_process:
                    self.logger.log(loss_dict)
            
            # Validation (only rank 0)
            if self.is_main_process and epoch % self.cfg.validation_freq == 0:
                val_metrics = self.validate()
                self.logger.log(val_metrics)
            
            # Save checkpoint (only rank 0)
            if self.is_main_process and epoch % self.cfg.save_freq == 0:
                self.save_checkpoint(epoch)
            
            # Synchronize all processes
            dist.barrier()
    
    def save_checkpoint(self, epoch):
        """保存checkpoint（只在rank 0）"""
        if not self.is_main_process:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.vla_actor.module.state_dict(),  # unwrap DDP
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.cfg.use_amp else None,
        }
        torch.save(checkpoint, f"{self.log_dir}/checkpoint_epoch_{epoch}.pth")
```

#### 方案B: DeepSpeed (推荐Phase 2，如果需要更大规模)

```python
# 使用DeepSpeed ZeRO-3进行更大规模训练
import deepspeed

@configclass
class DeepSpeedVLAPretrainRunnerCfg(VLAPretrainRunnerCfg):
    """DeepSpeed配置"""
    deepspeed_config: str = "configs/deepspeed/ds_config.json"


class DeepSpeedVLAPretrainRunner(VLAPretrainRunner):
    """使用DeepSpeed的分布式训练"""
    def __init__(self, cfg, log_dir, device):
        super().__init__(cfg, log_dir, device)
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.vla_actor,
            config=cfg.deepspeed_config,
        )
    
    # Similar to DDP but using DeepSpeed engine
```

### DeepSpeed Config Example

```json
{
  "train_batch_size": 256,
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

---

## 📁 更新的文件结构

### 新增分布式训练文件

```
source/RoboRenForce/RoboRenForce/
│
├── runners/vla/
│   ├── pretrain/
│   │   ├── __init__.py
│   │   ├── pretrain_runner.py                    # 单卡版本
│   │   ├── ✨ distributed_pretrain_runner.py    # 多卡DDP版本 ⭐
│   │   └── ✨ deepspeed_pretrain_runner.py      # DeepSpeed版本
│   │
│   ├── post_train/
│   │   ├── sft_runner.py                         # 单卡版本
│   │   └── ✨ distributed_sft_runner.py         # 多卡DDP版本 ⭐
│   │
│   └── rl/
│       └── rl_on_policy_runner.py                # RL暂时单卡（环境并行）
│
├── utils/
│   ├── ✨ distributed/                           # 分布式训练工具 ⭐
│   │   ├── __init__.py
│   │   ├── ddp_utils.py                          # DDP helper functions
│   │   ├── checkpoint_utils.py                   # 分布式checkpoint
│   │   ├── logging_utils.py                      # 分布式logging
│   │   └── launch_utils.py                       # Multi-process launcher
│   │
│   └── ✨ mixed_precision/                       # 混合精度训练
│       ├── __init__.py
│       └── amp_utils.py                          # AMP utilities
│
└── networks/
    ├── vlm/
    │   ├── vlm_base.py
    │   ├── qwen3_vl.py
    │   └── ✨ vlm_loader.py                      # 加载预训练VLM ⭐
    │
    └── action_heads/
        ├── ✨ action_expert.py                   # System 1核心 ⭐
        ├── diffusion_head.py
        └── regression_head.py
```

### 新增启动脚本

```
scripts/
├── renforce/                                      # 📝 低层RL脚本（现有）
│   ├── train_lab.py                              # Isaac Lab RL训练
│   ├── train_gym.py                              # Gym RL训练
│   └── play_lab.py                               # RL评估
│
├── ✨ vla/                                        # VLA专用脚本 ⭐ NEW
│   ├── __init__.py
│   │
│   ├── pretrain/                                 # VLA预训练
│   │   ├── train_single_gpu.py                  # 单卡预训练
│   │   ├── train_ddp.py                         # 多卡DDP预训练 ⭐
│   │   └── train_deepspeed.py                   # DeepSpeed预训练
│   │
│   ├── post_train/                               # VLA后训练
│   │   ├── train_sft.py                         # SFT训练
│   │   └── train_dpo.py                         # DPO训练
│   │
│   ├── rl/                                       # VLA RL微调
│   │   ├── train_ppo_finetune.py                # PPO微调
│   │   └── train_sac_finetune.py                # SAC微调
│   │
│   ├── eval/                                     # VLA评估
│   │   ├── evaluate_pretrain.py                 # 预训练模型评估
│   │   └── evaluate_rl.py                       # RL微调模型评估
│   │
│   └── utils/                                    # VLA工具脚本
│       ├── visualize_predictions.py             # 可视化预测
│       └── export_model.py                      # 模型导出
│
├── ✨ distributed/                                # 分布式启动工具（通用）
│   ├── __init__.py
│   ├── launch_ddp.py                            # DDP launcher
│   ├── launch_slurm.py                          # SLURM cluster launcher
│   └── launch_torchrun.py                       # torchrun wrapper
│
└── data/                                         # 数据处理脚本（现有）
    ├── rlds_to_lerobot.py
    └── ...
```

### 新增配置文件

```
configs/
├── ✨ distributed/
│   ├── ddp_config.yaml                           # DDP配置
│   ├── deepspeed_config.json                     # DeepSpeed配置
│   └── slurm_template.sh                         # SLURM脚本模板
│
└── ✨ system/
    ├── system2_vlm_configs.py                    # System 2预训练VLM列表
    ├── system1_action_expert_configs.py          # System 1配置
    └── system0_locomotion_configs.py             # System 0配置
```

---

## 🎯 重新设计的实现优先级

### Phase 0: 准备工作 (Week 0)
**目标**: 理解三层架构，准备分布式环境

```
✅ Tasks:
1. 阅读Psi0代码，理解三层system设计
2. 调研预训练VLM选择（Qwen3-VL vs OpenVLA vs ...）
3. 搭建多卡训练环境（8 GPUs）
4. 测试DDP基础功能

📁 Files: 0个（调研阶段）
```

### Phase 1: 数据 + VLM复用 (Week 1-2) ⭐⭐⭐
**目标**: 数据流打通 + 加载预训练VLM

```
✅ Priority P0 - Data Infrastructure:
1. dataset/lerobot/lerobot_dataset.py
2. dataset/lerobot/metadata.py
3. dataset/lerobot/stats_utils.py
4. utils/processor/processor_base.py
5. utils/processor/pipeline.py

✅ Priority P0 - VLM Loading:
6. networks/vlm/vlm_base.py
7. networks/vlm/qwen3_vl.py
8. networks/vlm/vlm_loader.py                     # ⭐ 加载预训练VLM

📁 Files: 8个
🎯 Outcome: 
   - 可以加载LeRobot数据
   - 可以加载预训练Qwen3-VL（冻结）
   - VLM forward pass工作
```

### Phase 2: System 1 Action Expert (Week 2-3) ⭐⭐⭐
**目标**: 实现Action Expert架构

```
✅ Priority P0 - Action Expert:
1. networks/action_heads/action_expert.py         # ⭐ System 1核心
2. networks/action_heads/diffusion_head.py
3. networks/action_heads/regression_head.py

✅ Priority P0 - VLA Actor:
4. components/actor/vla_actors.py                 # 三层组合
5. components/normalizer/image_normalizer.py

📁 Files: 5个
🎯 Outcome:
   - VLA Actor forward pass工作
   - System 2 (冻结VLM) + System 1 (Action Expert)
```

### Phase 3: 单卡训练验证 (Week 3) ⭐⭐⭐
**目标**: 单卡版本训练通，验证架构可行

```
✅ Priority P0 - Single-GPU Training:
1. algorithms/vla_pretrain/vla_pretrain_base.py
2. algorithms/losses/vla_losses.py
3. runners/vla/pretrain/pretrain_runner.py        # 单卡版本
4. scripts/renforce/train_vla_pretrain.py
5. demo_tasks/vla_pretrain/minimal_example.py

✅ Test:
6. tests/test_runners/test_vla_pretrain_runner.py

📁 Files: 6个
🎯 Outcome:
   - 单卡训练System 1成功
   - Action loss下降
   - 验证架构可行
```

### Phase 4: 分布式训练 (Week 4) ⭐⭐⭐
**目标**: 多卡DDP训练，加速System 1训练

```
✅ Priority P0 - DDP Infrastructure:
1. utils/distributed/ddp_utils.py
2. utils/distributed/checkpoint_utils.py
3. utils/distributed/logging_utils.py
4. utils/distributed/launch_utils.py

✅ Priority P0 - Distributed Runner:
5. runners/vla/pretrain/distributed_pretrain_runner.py  # ⭐ 多卡核心
6. scripts/renforce/train_vla_pretrain_distributed.py
7. scripts/distributed/launch_ddp.py

✅ Config:
8. configs/distributed/ddp_config.yaml

📁 Files: 8个
🎯 Outcome:
   - 8卡DDP训练System 1
   - 训练速度提升8x
   - Gradient sync正常
```

### Phase 5: Post-Train (Week 5) ⭐⭐
**目标**: SFT任务特化

```
✅ Priority P1 - SFT:
1. runners/vla/post_train/post_train_runner.py
2. runners/vla/post_train/sft_runner.py
3. runners/vla/post_train/distributed_sft_runner.py
4. scripts/renforce/train_vla_post_train.py

📁 Files: 4个
🎯 Outcome:
   - 在任务数据上SFT
   - Task success提升
```

### Phase 6: RL Fine-tune (Week 6) ⭐
**目标**: RL优化（环境并行）

```
✅ Priority P2 - RL:
1. algorithms/vla_pretrain/vla_rl_finetune.py
2. runners/vla/rl/rl_on_policy_runner.py
3. utils/env_wrapper/vla_wrapper/multimodal_env_wrapper.py
4. scripts/renforce/train_vla_rl_finetune.py

📁 Files: 4个
🎯 Outcome:
   - RL微调VLA
   - 环境并行（4096 envs）
   - 单卡learner + 多env workers
```

### Phase 7: System 0接口 (Week 7) ⭐
**目标**: 对接locomotion controller

```
✅ Priority P2 - Locomotion Interface:
1. components/locomotion/locomotion_tracker.py
2. utils/env_wrapper/vla_wrapper/hierarchical_wrapper.py

📁 Files: 2个
🎯 Outcome:
   - System 1 → System 0 tracking
   - 全身控制
```

---

## 📊 文件统计更新

### 核心新增文件

```
Phase 1 (Data + VLM):           8 files  ⭐⭐⭐
Phase 2 (Action Expert):        5 files  ⭐⭐⭐
Phase 3 (Single-GPU Training):  6 files  ⭐⭐⭐
Phase 4 (Distributed Training): 8 files  ⭐⭐⭐
Phase 5 (Post-Train):           4 files  ⭐⭐
Phase 6 (RL):                   4 files  ⭐
Phase 7 (System 0):             2 files  ⭐

Critical Path: Phase 1-4 (27 files, 4 weeks)
Total: ~37 核心文件
```

加上测试、文档、配置等，总计约**60-80个文件**（比之前的120个精简）

---

## 🚀 启动命令示例

### 单卡训练（Phase 3）

```bash
# System 1 Pretrain (单卡验证)
python scripts/renforce/train_vla_pretrain.py \
    --config demo_tasks/vla_pretrain/minimal_example.py \
    --data_root data/humanoid_demos \
    --log_dir logs/pretrain_single
```

### 多卡DDP训练（Phase 4）

```bash
# System 1 Pretrain (8卡DDP)
torchrun --nproc_per_node=8 \
    scripts/renforce/train_vla_pretrain_distributed.py \
    --config demo_tasks/vla_pretrain/humanoid_qwen3vl.py \
    --data_root data/mixed_robot_data \
    --log_dir logs/pretrain_ddp

# 或使用自定义launcher
python scripts/distributed/launch_ddp.py \
    --num_gpus 8 \
    --script scripts/renforce/train_vla_pretrain_distributed.py \
    --config demo_tasks/vla_pretrain/humanoid_qwen3vl.py
```

### SLURM集群训练（Phase 4+）

```bash
# 提交SLURM任务（64卡）
sbatch scripts/distributed/launch_slurm.sh \
    --nodes=8 \
    --gpus-per-node=8 \
    --config demo_tasks/vla_pretrain/large_scale.py
```

### RL Fine-tune（Phase 6）

```bash
# System 1 RL (环境并行，单卡learner)
python scripts/renforce/train_vla_rl_finetune.py \
    --task Isaac-Humanoid-Reach-v0 \
    --pretrained_vla checkpoints/pretrained_system1.pth \
    --num_envs 4096 \
    --log_dir logs/rl_finetune
```

---

## 🎓 关键设计决策总结

### ✅ 决策1: System 2复用，不从头训练
**理由**: 
- Qwen3-VL-2B已经在大规模ego数据上预训练
- 从头训练需要32-64 GPUs × 几周
- 冻结或LoRA微调足够

**实现**: `vlm_loader.py`加载HuggingFace模型

### ✅ 决策2: System 1是核心训练目标
**理由**:
- 机器人特定，必须训练
- 数据量可控（1k-10k trajectories）
- 8-16 GPUs × 几天可完成

**实现**: `action_expert.py` + `distributed_pretrain_runner.py`

### ✅ 决策3: DDP优先，DeepSpeed备选
**理由**:
- DDP更简单，PyTorch原生支持
- System 1模型不大（~500M params）
- 如果后续需要不冻结VLM，再用DeepSpeed

**实现**: Phase 4 DDP, Phase 8可选DeepSpeed

### ✅ 决策4: RL保持环境并行
**理由**:
- RL阶段模型已固定，主要瓶颈是环境
- 4096并行envs比数据并行更高效
- 单卡learner + 多env workers

**实现**: 复用现有`OnPolicyRunner`，扩展VLA支持

### ✅ 决策5: System 0假设已有
**理由**:
- RoboRenforce已有locomotion训练
- System 0是低层控制，与VLA解耦
- 只需要tracking接口

**实现**: Phase 7低优先级

---

## 📖 参考架构

- **Psi0**: 三层system设计
  - `.references/Psi0/src/psi/models/psi0.py`
- **LeRobot**: 数据格式 + 分布式训练
  - `.references/lerobot/src/lerobot/datasets/`
- **DeepSpeed**: 大规模训练
  - https://github.com/microsoft/DeepSpeed

---

这个架构更贴近实际需求，优先级清晰，关键是：
1. **复用System 2**（省资源）
2. **专注System 1**（核心价值）
3. **分布式必需**（多卡训练）
4. **渐进实现**（单卡→多卡→RL）
