# Scripts Organization - VLA vs RL Separation

**设计原则**: 清晰分离VLA和低层RL脚本，避免混淆

---

## 📁 新的Scripts目录结构

```
scripts/
│
├── renforce/                      # 📝 低层RL训练脚本（现有，不变）
│   ├── train_lab.py              # Isaac Lab RL训练
│   ├── train_gym.py              # Gym RL训练
│   └── play_lab.py               # RL策略评估
│
├── vla/                          # ✨ VLA专用脚本 (NEW) ⭐
│   ├── pretrain/                 # VLA预训练
│   │   ├── train_single_gpu.py  # 单卡预训练
│   │   ├── train_ddp.py         # 多卡DDP预训练 ⭐
│   │   └── train_deepspeed.py   # DeepSpeed预训练
│   │
│   ├── post_train/               # VLA后训练
│   │   ├── train_sft.py         # SFT训练
│   │   └── train_dpo.py         # DPO训练
│   │
│   ├── rl/                       # VLA RL微调
│   │   ├── train_ppo_finetune.py  # PPO微调
│   │   └── train_sac_finetune.py  # SAC微调
│   │
│   ├── eval/                     # VLA评估
│   │   ├── evaluate_pretrain.py  # 预训练评估
│   │   └── evaluate_rl.py       # RL微调评估
│   │
│   └── utils/                    # VLA工具
│       ├── visualize_predictions.py
│       └── export_model.py
│
├── distributed/                  # ✨ 分布式启动工具（通用）
│   ├── launch_ddp.py            # DDP launcher
│   ├── launch_slurm.py          # SLURM launcher
│   └── launch_torchrun.py       # torchrun wrapper
│
├── data/                         # 📝 数据处理脚本（现有）
│   ├── rlds_to_lerobot.py
│   ├── isaaclab_to_lerobot.py
│   └── ...
│
└── third_party/                  # 📝 第三方脚本（现有）
```

---

## 🎯 设计理念

### 为什么分离？

#### ❌ 旧设计（混合）
```
scripts/renforce/
├── train_lab.py                  # RL训练
├── train_gym.py                  # RL训练
├── train_vla_pretrain.py         # VLA预训练 ← 混在一起
├── train_vla_rl_finetune.py      # VLA RL ← 混在一起
└── play_lab.py                   # RL评估

问题:
- VLA和RL脚本混在一起
- 不清楚哪个是高层VLA，哪个是低层RL
- 扩展性差（VLA脚本越来越多会很乱）
```

#### ✅ 新设计（分离）
```
scripts/
├── renforce/        # 低层RL：PPO, SAC locomotion训练
├── vla/            # 高层VLA：System 1训练 + RL微调
└── distributed/    # 通用分布式工具

优点:
- 清晰的层次结构
- VLA脚本独立目录，易于扩展
- 低层RL保持不变
- 分布式工具可被两者复用
```

---

## 📝 脚本文件清单

### VLA脚本（15个文件）

#### Pretrain (3个)
```
scripts/vla/pretrain/
├── train_single_gpu.py          # 单卡预训练（Week 3验证）
├── train_ddp.py                 # 多卡DDP预训练（Week 4核心）⭐
└── train_deepspeed.py           # DeepSpeed预训练（可选）
```

#### Post-Train (2个)
```
scripts/vla/post_train/
├── train_sft.py                 # SFT训练（Week 5）
└── train_dpo.py                 # DPO训练（未来）
```

#### RL Fine-tune (2个)
```
scripts/vla/rl/
├── train_ppo_finetune.py        # PPO微调（Week 6）
└── train_sac_finetune.py        # SAC微调（可选）
```

#### Evaluation (2个)
```
scripts/vla/eval/
├── evaluate_pretrain.py         # 预训练模型评估
└── evaluate_rl.py               # RL微调模型评估
```

#### Utils (2个)
```
scripts/vla/utils/
├── visualize_predictions.py     # 可视化VLA预测
└── export_model.py              # 模型导出（ONNX/TorchScript）
```

#### 加上__init__.py (4个)
```
scripts/vla/__init__.py
scripts/vla/pretrain/__init__.py
scripts/vla/post_train/__init__.py
scripts/vla/rl/__init__.py
scripts/vla/eval/__init__.py
scripts/vla/utils/__init__.py
```

**VLA脚本总计**: 15 + 6 = 21个文件

### 分布式工具（4个文件）
```
scripts/distributed/
├── __init__.py
├── launch_ddp.py                # DDP launcher helper
├── launch_slurm.py              # SLURM job generator
└── launch_torchrun.py           # torchrun wrapper
```

### 数据脚本（6个文件，现有）
```
scripts/data/
├── __init__.py
├── rlds_to_lerobot.py
├── isaaclab_to_lerobot.py
├── compute_dataset_stats.py
├── merge_datasets.py
└── visualize_dataset.py
```

**总计新增脚本**: 21 (VLA) + 4 (distributed) = 25个文件

---

## 🚀 使用示例

### Phase 3: 单卡训练验证

```bash
# 单卡预训练
python scripts/vla/pretrain/train_single_gpu.py \
    --config demo_tasks/vla_pretrain/minimal_example.py \
    --data_root data/humanoid_demos \
    --log_dir logs/pretrain_single
```

### Phase 4: 多卡DDP训练 ⭐

```bash
# 方式1: 直接使用torchrun
torchrun --nproc_per_node=8 \
    scripts/vla/pretrain/train_ddp.py \
    --config demo_tasks/vla_pretrain/humanoid_qwen3vl.py \
    --data_root data/mixed_robot_data \
    --log_dir logs/pretrain_ddp

# 方式2: 使用launcher helper
python scripts/distributed/launch_ddp.py \
    --num_gpus 8 \
    --script scripts/vla/pretrain/train_ddp.py \
    -- --config demo_tasks/vla_pretrain/humanoid_qwen3vl.py

# 方式3: SLURM集群（64卡）
python scripts/distributed/launch_slurm.py \
    --nodes 8 \
    --gpus_per_node 8 \
    --script scripts/vla/pretrain/train_ddp.py \
    --config demo_tasks/vla_pretrain/large_scale.py
```

### Phase 5: SFT后训练

```bash
# 4卡DDP SFT
torchrun --nproc_per_node=4 \
    scripts/vla/post_train/train_sft.py \
    --pretrained checkpoints/pretrained_system1.pth \
    --data_root data/task_specific_demos \
    --log_dir logs/sft
```

### Phase 6: RL微调

```bash
# PPO微调（环境并行，单卡learner）
python scripts/vla/rl/train_ppo_finetune.py \
    --task Isaac-Humanoid-Reach-v0 \
    --pretrained checkpoints/sft_system1.pth \
    --num_envs 4096 \
    --log_dir logs/rl_finetune
```

### 评估

```bash
# 评估预训练模型
python scripts/vla/eval/evaluate_pretrain.py \
    --vla_path checkpoints/pretrained_vla.pth \
    --data_root data/validation_set \
    --output_dir results/pretrain_eval

# 评估RL微调模型
python scripts/vla/eval/evaluate_rl.py \
    --vla_path checkpoints/rl_finetuned_vla.pth \
    --task Isaac-Humanoid-Reach-v0 \
    --num_episodes 100
```

### 可视化

```bash
# 可视化VLA预测
python scripts/vla/utils/visualize_predictions.py \
    --vla_path checkpoints/vla.pth \
    --data_root data/demo_dataset \
    --output_dir visualizations
```

### 模型导出

```bash
# 导出为ONNX
python scripts/vla/utils/export_model.py \
    --vla_path checkpoints/vla.pth \
    --format onnx \
    --output exported_vla.onnx
```

---

## 🔄 与低层RL的关系

### 低层RL脚本（不变）
```bash
# 这些脚本保持不变，用于训练System 0 (locomotion)
python scripts/renforce/train_lab.py --task Isaac-Humanoid-v0 ...
python scripts/renforce/train_gym.py --env HalfCheetah-v4 ...
python scripts/renforce/play_lab.py --task Isaac-Humanoid-v0 ...
```

### VLA脚本（新增）
```bash
# 这些脚本用于训练System 1 (Action Expert)
python scripts/vla/pretrain/train_ddp.py ...
python scripts/vla/rl/train_ppo_finetune.py ...
```

### 分布式工具（共享）
```bash
# 这些工具可以被VLA和RL共享
python scripts/distributed/launch_ddp.py --script scripts/vla/pretrain/train_ddp.py ...
python scripts/distributed/launch_ddp.py --script scripts/renforce/train_lab.py ...  # 如果需要
```

---

## 📊 实施优先级

### Phase 3 (Week 3)
```
Priority P0:
✅ scripts/vla/pretrain/train_single_gpu.py  # 单卡验证
```

### Phase 4 (Week 4) ⭐ 核心
```
Priority P0:
✅ scripts/vla/pretrain/train_ddp.py         # 多卡DDP
✅ scripts/distributed/launch_ddp.py          # DDP launcher
✅ scripts/distributed/launch_torchrun.py     # torchrun wrapper

Priority P1:
✅ scripts/distributed/launch_slurm.py        # SLURM支持（集群环境）
```

### Phase 5 (Week 5)
```
Priority P1:
✅ scripts/vla/post_train/train_sft.py       # SFT训练
```

### Phase 6 (Week 6)
```
Priority P1:
✅ scripts/vla/rl/train_ppo_finetune.py      # PPO微调
```

### Phase 7+ (可选)
```
Priority P2:
✅ scripts/vla/pretrain/train_deepspeed.py   # DeepSpeed（大规模）
✅ scripts/vla/post_train/train_dpo.py       # DPO（未来）
✅ scripts/vla/rl/train_sac_finetune.py      # SAC微调
✅ scripts/vla/eval/evaluate_pretrain.py     # 评估工具
✅ scripts/vla/eval/evaluate_rl.py
✅ scripts/vla/utils/visualize_predictions.py
✅ scripts/vla/utils/export_model.py
```

---

## 🎯 脚本模板

### train_ddp.py 模板
```python
"""
VLA Pretrain with DDP
Usage:
    torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
        --config demo_tasks/vla_pretrain/humanoid_qwen3vl.py
"""
import argparse
import torch.distributed as dist
from RoboRenForce.runners.vla.pretrain import DistributedVLAPretrainRunner

def main(args):
    # 1. Setup distributed
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    # 2. Load config
    config = load_config(args.config)
    
    # 3. Create runner
    runner = DistributedVLAPretrainRunner(
        config,
        log_dir=args.log_dir,
        device=f'cuda:{local_rank}'
    )
    
    # 4. Train
    runner.learn(num_epochs=config.num_epochs)
    
    # 5. Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--log_dir', type=str, default='logs/pretrain_ddp')
    args = parser.parse_args()
    
    main(args)
```

---

## 📁 完整文件树（更新）

```
scripts/
├── renforce/                      # 低层RL（3个文件，现有）
│   ├── train_lab.py
│   ├── train_gym.py
│   └── play_lab.py
│
├── vla/                          # VLA（21个文件，新增）⭐
│   ├── __init__.py
│   ├── pretrain/
│   │   ├── __init__.py
│   │   ├── train_single_gpu.py
│   │   ├── train_ddp.py         ⭐⭐⭐
│   │   └── train_deepspeed.py
│   ├── post_train/
│   │   ├── __init__.py
│   │   ├── train_sft.py
│   │   └── train_dpo.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── train_ppo_finetune.py
│   │   └── train_sac_finetune.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── evaluate_pretrain.py
│   │   └── evaluate_rl.py
│   └── utils/
│       ├── __init__.py
│       ├── visualize_predictions.py
│       └── export_model.py
│
├── distributed/                  # 分布式工具（4个文件，新增）
│   ├── __init__.py
│   ├── launch_ddp.py            ⭐
│   ├── launch_slurm.py
│   └── launch_torchrun.py
│
├── data/                         # 数据脚本（6个文件，现有）
│   ├── __init__.py
│   ├── rlds_to_lerobot.py
│   ├── isaaclab_to_lerobot.py
│   ├── compute_dataset_stats.py
│   ├── merge_datasets.py
│   └── visualize_dataset.py
│
└── third_party/                  # 第三方脚本（现有）
```

**新增脚本总计**: 25个文件
**核心脚本（P0）**: 2个（train_ddp.py + launch_ddp.py）

---

## ✅ 优势总结

1. **清晰分离**: VLA和RL脚本完全分开
2. **易于扩展**: VLA脚本在独立目录，方便添加新功能
3. **复用工具**: 分布式工具可被VLA和RL共享
4. **一致命名**: `train_*.py`, `evaluate_*.py`, `launch_*.py`
5. **层次清晰**: pretrain → post_train → rl，对应训练阶段

这个设计更符合"高层VLA vs 低层RL"的概念分离！
