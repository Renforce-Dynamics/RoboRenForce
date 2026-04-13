#!/usr/bin/env python3
"""
生成小型合成数据集用于测试

生成LeRobot Parquet格式的dummy数据集，用于快速测试VLA训练pipeline。

Usage:
    python scripts/data/generate_dummy_dataset.py \
        --output_dir data/dummy_dataset \
        --num_episodes 10 \
        --episode_length 50

生成内容:
- Parquet episode files (train/episode_*.parquet)
- Normalization stats (meta/stats.safetensors)
- Dataset info (meta/info.json)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import save_file
import json


def generate_dummy_dataset(
    output_dir: str,
    num_episodes: int = 10,
    episode_length: int = 50,
    action_dim: int = 19,
    proprio_dim: int = 48,
):
    """
    生成小型dummy数据集。

    Args:
        output_dir: 输出目录
        num_episodes: Episode数量
        episode_length: 每个episode的长度
        action_dim: 动作维度 (默认19, humanoid)
        proprio_dim: 本体感知维度 (默认48)
    """
    output_path = Path(output_dir)

    # 创建目录结构
    (output_path / "meta").mkdir(parents=True, exist_ok=True)
    (output_path / "train").mkdir(exist_ok=True)
    (output_path / "train/videos").mkdir(exist_ok=True)

    print(f"Generating {num_episodes} dummy episodes...")
    print(f"Episode length: {episode_length}")
    print(f"Action dim: {action_dim}, Proprio dim: {proprio_dim}")

    # 生成episodes
    total_frames = 0
    for ep_idx in range(num_episodes):
        # 生成数据
        data = {
            "episode_index": [ep_idx] * episode_length,
            "frame_index": list(range(episode_length)),
            "timestamp": np.linspace(0, 5, episode_length),  # 5秒

            # 动作: 随机但连续 (添加一些平滑性)
            "action": np.cumsum(
                np.random.randn(episode_length, action_dim) * 0.1,
                axis=0
            ).astype(np.float32),

            # 本体感知: 随机但连续
            "proprioception": np.cumsum(
                np.random.randn(episode_length, proprio_dim) * 0.05,
                axis=0
            ).astype(np.float32),

            # 奖励: 随机，最后一帧高奖励
            "reward": np.concatenate([
                np.random.rand(episode_length - 1) * 0.1,
                [1.0]  # 最后一帧成功
            ]).astype(np.float32),

            # Done标志
            "done": [False] * (episode_length - 1) + [True],

            # 图像路径 (指向虚拟视频文件)
            "observation.image": [
                f"videos/episode_{ep_idx:06d}_camera_0.mp4"
            ] * episode_length,
        }

        # 保存为Parquet
        df = pd.DataFrame(data)
        parquet_path = output_path / "train" / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path, index=False)

        total_frames += episode_length

        if (ep_idx + 1) % 10 == 0:
            print(f"  Generated {ep_idx + 1}/{num_episodes} episodes...")

    print(f"✓ Generated {num_episodes} episodes ({total_frames} total frames)")

    # 生成stats.safetensors (归一化统计)
    print("Generating normalization stats...")
    stats = {
        "action.mean": np.zeros(action_dim, dtype=np.float32),
        "action.std": np.ones(action_dim, dtype=np.float32),
        "proprioception.mean": np.zeros(proprio_dim, dtype=np.float32),
        "proprioception.std": np.ones(proprio_dim, dtype=np.float32),
    }
    stats_path = output_path / "meta/stats.safetensors"
    save_file(stats, stats_path)
    print(f"✓ Generated {stats_path}")

    # 生成info.json (数据集元数据)
    print("Generating dataset metadata...")
    info = {
        "fps": 10,
        "robot_type": "humanoid_dummy",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "action_dim": action_dim,
        "proprioception_dim": proprio_dim,
        "image_size": [224, 224],
        "cameras": ["camera_0"],
        "video_codec": "dummy",  # 没有真实视频
    }
    info_path = output_path / "meta/info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"✓ Generated {info_path}")

    # 打印总结
    print("\n" + "=" * 60)
    print("✓ Dummy dataset created successfully!")
    print("=" * 60)
    print(f"Location: {output_dir}")
    print(f"Episodes: {num_episodes}")
    print(f"Total frames: {total_frames}")
    print(f"Action dim: {action_dim}")
    print(f"Proprio dim: {proprio_dim}")
    print("\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── meta/")
    print(f"    │   ├── stats.safetensors")
    print(f"    │   └── info.json")
    print(f"    └── train/")
    print(f"        ├── episode_000000.parquet")
    print(f"        ├── episode_000001.parquet")
    print(f"        └── ...")
    print("\nNext steps:")
    print("  1. Verify dataset:")
    print(f"     ls -lh {output_dir}/train/")
    print("  2. Test loading:")
    print("     python -c 'from RoboRenForce.dataset.lerobot import LeRobotDataset; ...'")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dummy LeRobot dataset for testing"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/dummy_dataset",
        help="Output directory for dummy dataset",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to generate",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=50,
        help="Length of each episode (timesteps)",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=19,
        help="Action dimension (default: 19 for humanoid)",
    )
    parser.add_argument(
        "--proprio_dim",
        type=int,
        default=48,
        help="Proprioception dimension (default: 48)",
    )

    args = parser.parse_args()

    generate_dummy_dataset(
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
    )


if __name__ == "__main__":
    main()
