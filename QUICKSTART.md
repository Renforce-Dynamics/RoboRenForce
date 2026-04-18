# VLA Training Pipeline 快速开始

**目标**: 从数据准备到单卡训练的最快路径

---

## 🚀 立即开始 (5分钟)

### Step 1: 安装依赖

```bash
# 核心依赖
pip install pandas pyarrow safetensors

# (可选) 下载真实数据集
pip install huggingface_hub
```

### Step 2: 生成测试数据集

```bash
# 创建数据目录
mkdir -p data

# 生成10个episode的dummy数据集 (1秒完成)
python scripts/data/generate_dummy_dataset.py \
    --output_dir data/dummy_dataset \
    --num_episodes 10

# 验证
ls -lh data/dummy_dataset/train/
```

**输出**:
```
✓ Dummy dataset (LeRobot v2) created successfully!
Location: data/dummy_dataset
Episodes: 10
Total frames: 500
Format: LeRobot v2 (chunk-based)
```

### Step 3 (可选): 下载真实数据集

```bash
# 下载Push-T数据集 (小型, ~2GB, 5-10分钟)
python scripts/data/download_lerobot_dataset.py \
    --repo_id lerobot/pusht \
    --output_dir data/lerobot/pusht

# 或使用CLI (更简单)
huggingface-cli download lerobot/pusht \
    --repo-type dataset \
    --local-dir data/lerobot/pusht
```

---

## 📊 推荐的数据集选择

### 方案A: Dummy数据集 (本周推荐) ⭐

**优点**:
- ✅ 1秒生成
- ✅ 体积小 (<1MB)
- ✅ 立即开始测试Phase 1-3

**缺点**:
- ❌ 没有真实图像 (仅路径)
- ❌ 无法测试视频加载

**使用场景**: 快速验证pipeline (Phase 1-3)

```bash
python scripts/data/generate_dummy_dataset.py
```

### 方案B: Push-T数据集 (下周推荐)

**优点**:
- ✅ 真实数据
- ✅ 包含视频
- ✅ LeRobot官方数据集

**缺点**:
- ❌ 下载需要5-10分钟
- ❌ 体积较大 (~2GB)

**使用场景**: Phase 1-4完整测试

```bash
python scripts/data/download_lerobot_dataset.py \
    --repo_id lerobot/pusht \
    --output_dir data/lerobot/pusht
```

### 方案C: ALOHA Sim数据集

**优点**:
- ✅ 多相机
- ✅ 接近真实机器人格式

**缺点**:
- ❌ 下载较慢 (~1GB)

```bash
python scripts/data/download_lerobot_dataset.py \
    --repo_id lerobot/aloha_sim_insertion_human \
    --output_dir data/lerobot/aloha_sim
```

---

## 🎯 本周计划 (Week 0-1)

### Today: 数据准备 ✅

```bash
# 1. 生成dummy数据集
python scripts/data/generate_dummy_dataset.py

# 2. (可选) 下载Push-T
python scripts/data/download_lerobot_dataset.py --repo_id lerobot/pusht
```

### This Week: 实现Phase 1 (数据加载 + VLM)

**优先级顺序**:
1. ✅ 准备数据集 (今天完成)
2. ⏳ 实现 `dataset/lerobot/lerobot_dataset.py`
3. ⏳ 实现 `dataset/lerobot/lerobot_processor.py`
4. ⏳ 实现 `networks/vlm/qwen3vl.py`

**测试目标**:
```python
# 能够成功运行:
from RoboRenForce.dataset.lerobot import LeRobotDatasetCfg

cfg = LeRobotDatasetCfg(data_root="data/dummy_dataset", split="train")
dataset = cfg.construct_from_cfg()
print(f"Dataset size: {len(dataset)}")

sample = dataset[0]
print(f"Keys: {sample.keys()}")
print(f"Action: {sample['action'].shape}")
```

---

## 📁 数据集格式说明

### LeRobot Parquet格式

```
data/dummy_dataset/
├── meta/
│   ├── stats.safetensors    # 归一化统计 (mean/std)
│   └── info.json           # 数据集元数据
└── train/
    ├── episode_000000.parquet  # Episode 0
    ├── episode_000001.parquet  # Episode 1
    └── ...
```

### Parquet文件内容

每个episode文件包含:
- `episode_index`: Episode ID
- `frame_index`: 帧索引
- `timestamp`: 时间戳
- `action`: (T, action_dim) 动作
- `proprioception`: (T, proprio_dim) 本体感知
- `reward`: (T,) 奖励
- `done`: (T,) Done标志
- `observation.image`: 图像路径

### stats.safetensors内容

```python
{
    "action.mean": [19,],      # 动作均值
    "action.std": [19,],       # 动作标准差
    "proprioception.mean": [48,],
    "proprioception.std": [48,],
}
```

---

## 🧪 验证数据集

### 验证Parquet文件

```python
import pandas as pd

# 读取一个episode
df = pd.read_parquet("data/dummy_dataset/train/episode_000000.parquet")

print(f"Episode length: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Action shape: {df['action'].iloc[0].shape}")
```

### 验证stats文件

```python
from safetensors.numpy import load_file

stats = load_file("data/dummy_dataset/meta/stats.safetensors")

print(f"Keys: {stats.keys()}")
print(f"Action mean shape: {stats['action.mean'].shape}")
```

---

## 🔧 调试技巧

### 问题1: 缺少依赖

```bash
# 安装所有需要的包
pip install pandas pyarrow safetensors huggingface_hub
```

### 问题2: 数据集路径错误

```python
# 使用绝对路径
from pathlib import Path
data_root = Path("data/dummy_dataset").absolute()
print(f"Using: {data_root}")
```

### 问题3: HuggingFace下载慢

```bash
# 使用镜像 (中国用户)
export HF_ENDPOINT=https://hf-mirror.com

# 然后下载
huggingface-cli download lerobot/pusht --repo-type dataset
```

---

## 📖 相关文档

- **数据集准备详细指南**: [.claude/DATASET-PREPARATION.md](.claude/DATASET-PREPARATION.md)
- **P0文件实施清单**: [P0-IMPLEMENTATION-CHECKLIST.md](P0-IMPLEMENTATION-CHECKLIST.md)
- **Phase 1实施细节**: [.claude/project-structure-core.md](.claude/project-structure-core.md) Section 4

---

## ✅ 检查清单

今天完成:
- [x] 创建数据目录
- [x] 生成dummy数据集
- [x] 验证数据集结构
- [ ] (可选) 下载Push-T数据集

本周完成:
- [ ] 实现LeRobotDataset
- [ ] 实现LeRobotProcessor
- [ ] 实现Qwen3VL
- [ ] 测试数据加载pipeline

---

**现状**: ✅ 数据集准备就绪  
**下一步**: 开始实现Phase 1 (数据加载)  
**预计时间**: Week 1
