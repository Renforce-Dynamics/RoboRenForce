# 数据集准备指南

**目标**: 准备小型测试数据集用于VLA训练pipeline验证

---

## 推荐的测试数据集

### 选项1: LeRobot Hub 数据集 (推荐) ⭐

LeRobot在HuggingFace Hub上提供了多个预处理好的数据集，已经是Parquet格式。

#### 推荐的小数据集：

**1. `lerobot/pusht` - Push-T 任务 (最简单)**
- 大小: ~500 episodes, ~2GB
- 任务: 2D推箱子任务
- 观测: 单相机RGB图像 + 2D位置
- 动作: 2D位置控制
- 优点: 最简单，最快下载

**2. `lerobot/aloha_sim_insertion_human` - ALOHA仿真 (中等)**
- 大小: ~50 episodes, ~1GB
- 任务: 双臂插入任务
- 观测: 多相机 + 关节位置
- 动作: 14维关节控制
- 优点: 接近真实机器人数据格式

**3. `lerobot/xarm_lift_medium` - X-Arm 举起任务 (真实机器人)**
- 大小: ~200 episodes, ~3GB
- 任务: 物体抓取举起
- 观测: 相机 + 本体感知
- 动作: 6D机械臂控制

---

## 方法1: 直接从LeRobot Hub下载 (最简单) ⭐

### Step 1: 安装LeRobot (可选，仅用于下载)

```bash
# 临时安装LeRobot用于下载数据集
pip install lerobot
```

### Step 2: 下载数据集

创建下载脚本：

```python
# scripts/data/download_lerobot_dataset.py
"""
从LeRobot Hub下载数据集

Usage:
    python scripts/data/download_lerobot_dataset.py \
        --repo_id lerobot/pusht \
        --output_dir data/lerobot/pusht
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def download_dataset(repo_id: str, output_dir: str):
    """
    从HuggingFace Hub下载LeRobot数据集
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "lerobot/pusht")
        output_dir: 本地保存路径
    """
    print(f"Downloading {repo_id} to {output_dir}...")
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    
    print(f"Downloaded to {output_dir}")
    
    # 验证数据集结构
    output_path = Path(output_dir)
    
    if (output_path / "meta").exists():
        print(f"✓ Found meta/ directory")
    if (output_path / "train").exists():
        print(f"✓ Found train/ directory")
    if (output_path / "data").exists():
        print(f"✓ Found data/ directory (LeRobot v2 format)")
    
    print("\nDataset structure:")
    for item in output_path.iterdir():
        print(f"  - {item.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="lerobot/pusht")
    parser.add_argument("--output_dir", type=str, default="data/lerobot/pusht")
    args = parser.parse_args()
    
    download_dataset(args.repo_id, args.output_dir)
```

### Step 3: 运行下载

```bash
# 创建数据目录
mkdir -p data/lerobot

# 下载Push-T数据集 (最小，推荐测试用)
python scripts/data/download_lerobot_dataset.py \
    --repo_id lerobot/pusht \
    --output_dir data/lerobot/pusht

# 或者下载ALOHA仿真数据集
python scripts/data/download_lerobot_dataset.py \
    --repo_id lerobot/aloha_sim_insertion_human \
    --output_dir data/lerobot/aloha_sim
```

---

## 方法2: 使用HuggingFace CLI (更简单)

```bash
# 安装HuggingFace CLI
pip install huggingface_hub

# 下载数据集
huggingface-cli download \
    lerobot/pusht \
    --repo-type dataset \
    --local-dir data/lerobot/pusht

# 查看下载的文件
ls -lh data/lerobot/pusht/
```

---

## 方法3: 生成小型合成数据集 (最快测试)

如果只是想快速测试pipeline，可以生成一个超小的合成数据集：

```python
# scripts/data/generate_dummy_dataset.py
"""
生成小型合成数据集用于测试

生成格式: LeRobot Parquet格式
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import json
from safetensors.numpy import save_file


def generate_dummy_dataset(output_dir: str, num_episodes: int = 10):
    """
    生成小型dummy数据集
    
    格式:
    - 10个episodes
    - 每个episode 50 timesteps
    - 单相机 224x224 RGB
    - 19维动作 (类似humanoid)
    """
    output_path = Path(output_dir)
    
    # 创建目录结构
    (output_path / "meta").mkdir(parents=True, exist_ok=True)
    (output_path / "train").mkdir(exist_ok=True)
    (output_path / "train/videos").mkdir(exist_ok=True)
    
    print(f"Generating {num_episodes} dummy episodes...")
    
    # 生成episodes
    for ep_idx in range(num_episodes):
        ep_length = 50
        
        # 生成数据
        data = {
            "episode_index": [ep_idx] * ep_length,
            "frame_index": list(range(ep_length)),
            "timestamp": np.linspace(0, 5, ep_length),  # 5秒
            "action": np.random.randn(ep_length, 19).astype(np.float32),
            "proprioception": np.random.randn(ep_length, 48).astype(np.float32),
            "reward": np.random.rand(ep_length).astype(np.float32),
            "done": [False] * (ep_length - 1) + [True],
            # 图像路径 (实际不生成视频，只是路径)
            "observation.image": [
                f"videos/episode_{ep_idx:06d}_camera_0.mp4"
            ] * ep_length,
        }
        
        # 保存为Parquet
        df = pd.DataFrame(data)
        parquet_path = output_path / "train" / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path, index=False)
    
    print(f"Generated {num_episodes} episodes")
    
    # 生成stats.safetensors
    stats = {
        "action.mean": np.zeros(19, dtype=np.float32),
        "action.std": np.ones(19, dtype=np.float32),
        "proprioception.mean": np.zeros(48, dtype=np.float32),
        "proprioception.std": np.ones(48, dtype=np.float32),
    }
    save_file(stats, output_path / "meta/stats.safetensors")
    print("Generated stats.safetensors")
    
    # 生成info.json
    info = {
        "fps": 10,
        "robot_type": "humanoid",
        "total_episodes": num_episodes,
        "total_frames": num_episodes * 50,
        "action_dim": 19,
        "proprioception_dim": 48,
        "image_size": [224, 224],
    }
    with open(output_path / "meta/info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("Generated info.json")
    
    print(f"\n✓ Dummy dataset created at {output_dir}")
    print(f"  - {num_episodes} episodes")
    print(f"  - {num_episodes * 50} total frames")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/dummy_dataset")
    parser.add_argument("--num_episodes", type=int, default=10)
    args = parser.parse_args()
    
    generate_dummy_dataset(args.output_dir, args.num_episodes)
```

使用：
```bash
# 生成10个episodes的dummy数据集
python scripts/data/generate_dummy_dataset.py \
    --output_dir data/dummy_dataset \
    --num_episodes 10

# 验证
ls -lh data/dummy_dataset/
```

---

## 推荐的测试流程

### Phase 0 测试 (本周)

**选项A: 使用dummy数据集 (最快)**
```bash
# 1. 生成dummy数据集
python scripts/data/generate_dummy_dataset.py --num_episodes 10

# 2. 跳过数据转换，直接测试Phase 1 (数据加载)
```

**选项B: 下载LeRobot数据集 (真实数据)**
```bash
# 1. 下载Push-T (最小)
huggingface-cli download lerobot/pusht \
    --repo-type dataset \
    --local-dir data/lerobot/pusht

# 2. 验证数据集结构
ls -lh data/lerobot/pusht/

# 3. 跳过数据转换，直接测试Phase 1 (数据加载)
```

**选项C: 测试数据转换 (完整流程，推荐用于最后验证)**
```bash
# 需要先有Isaac Lab demos或RLDS数据
# 稍后在Phase 0实现完成后测试
```

### Phase 1 测试 (下周)

使用上面下载/生成的数据集测试：
```bash
# 测试数据集加载
python -c "
from RoboRenForce.dataset.lerobot import LeRobotDatasetCfg

cfg = LeRobotDatasetCfg(
    data_root='data/lerobot/pusht',  # 或 'data/dummy_dataset'
    split='train',
)

dataset = cfg.construct_from_cfg()
print(f'Dataset size: {len(dataset)}')

sample = dataset[0]
print(f'Sample keys: {sample.keys()}')
print(f'Image shape: {sample[\"image\"].shape}')
print(f'Action shape: {sample[\"action\"].shape}')
"
```

---

## 数据集大小对比

| 数据集 | Episodes | 大小 | 下载时间 | 用途 |
|--------|----------|------|----------|------|
| **Dummy生成** | 10 | <1MB | 1秒 | 快速测试pipeline |
| **lerobot/pusht** | 500 | ~2GB | 5-10分钟 | 真实数据测试 |
| **lerobot/aloha_sim** | 50 | ~1GB | 3-5分钟 | 多模态测试 |
| **lerobot/xarm_lift** | 200 | ~3GB | 10-15分钟 | 真实机器人测试 |

---

## 推荐方案 ⭐

**本周 (Phase 0 准备)**:
1. ✅ 先生成dummy数据集 (1分钟)
   ```bash
   python scripts/data/generate_dummy_dataset.py
   ```

2. ✅ 然后下载Push-T数据集 (10分钟)
   ```bash
   huggingface-cli download lerobot/pusht \
       --repo-type dataset \
       --local-dir data/lerobot/pusht
   ```

3. ✅ 暂时跳过Phase 0的数据转换实现，直接进入Phase 1测试数据加载

**原因**:
- Dummy数据集可以立即测试Phase 1-3的实现
- Push-T是真实数据，验证pipeline正确性
- Phase 0的数据转换可以放到最后实现（用于转换Isaac Lab自己的数据）

**下周开始实现时**:
- Week 1: Phase 1 (用dummy或pusht测试)
- Week 2: Phase 2 (继续用同样数据)
- Week 3: Phase 3 (单卡训练测试)
- Week 4: Phase 4 (DDP测试)
- 最后回来实现Phase 0 (如果需要转换Isaac Lab数据)

---

## 快速开始命令

```bash
# 1. 创建数据目录
mkdir -p data/{lerobot,dummy_dataset}

# 2. 生成dummy数据集 (测试用)
python scripts/data/generate_dummy_dataset.py \
    --output_dir data/dummy_dataset \
    --num_episodes 10

# 3. (可选) 下载真实数据集
pip install huggingface_hub
huggingface-cli download lerobot/pusht \
    --repo-type dataset \
    --local-dir data/lerobot/pusht

# 4. 验证
ls -lh data/dummy_dataset/
ls -lh data/lerobot/pusht/
```

---

**建议**: 现在创建dummy数据集生成脚本，然后直接开始实现Phase 1的数据集加载！
