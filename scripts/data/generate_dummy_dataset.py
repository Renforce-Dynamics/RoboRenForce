#!/usr/bin/env python3
"""
生成小型合成数据集用于测试

生成LeRobot v2格式的dummy数据集，用于快速测试VLA训练pipeline。

Usage:
    python scripts/data/generate_dummy_dataset.py \
        --output_dir data/dummy_dataset \
        --num_episodes 10 \
        --episode_length 50

生成内容:
- Chunk-based Parquet files (data/chunk-*/file-*.parquet)
- Normalization stats (meta/stats.json)
- Dataset info (meta/info.json)
- Episode metadata (meta/episodes/)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import json


def generate_dummy_dataset_v2(
    output_dir: str,
    num_episodes: int = 10,
    episode_length: int = 50,
    action_dim: int = 19,
    proprio_dim: int = 48,
    chunks_size: int = 1000,
    fps: int = 10,
):
    """
    生成LeRobot v2格式的dummy数据集。

    Args:
        output_dir: 输出目录
        num_episodes: Episode数量
        episode_length: 每个episode的长度
        action_dim: 动作维度 (默认19, humanoid)
        proprio_dim: 本体感知维度 (默认48)
        chunks_size: 每个chunk的最大帧数
        fps: 帧率
    """
    output_path = Path(output_dir)

    # 创建目录结构 (v2格式)
    (output_path / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "observation.image" / "chunk-000").mkdir(
        parents=True, exist_ok=True
    )

    print(f"Generating {num_episodes} dummy episodes (LeRobot v2 format)...")
    print(f"Episode length: {episode_length}")
    print(f"Action dim: {action_dim}, Proprio dim: {proprio_dim}")
    print(f"Chunk size: {chunks_size}")

    # 收集所有帧数据
    all_frames = []
    total_frames = 0

    for ep_idx in range(num_episodes):
        # 生成本体感知数据 (随机但连续)
        proprioceptions = np.cumsum(
            np.random.randn(episode_length, proprio_dim) * 0.05, axis=0
        ).astype(np.float32)

        # 生成动作数据 (随机但连续)
        actions = np.cumsum(
            np.random.randn(episode_length, action_dim) * 0.1, axis=0
        ).astype(np.float32)

        # 为每一帧构建数据
        for frame_idx in range(episode_length):
            frame_data = {
                # 本体感知 (observation.state)
                "observation.state": proprioceptions[frame_idx].tolist(),
                # 动作
                "action": actions[frame_idx].tolist(),
                # Episode和帧索引
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                # 时间戳 (float32)
                "timestamp": np.float32(frame_idx / fps),
                # Next step信息 (float32)
                "next.reward": np.float32(
                    1.0 if frame_idx == episode_length - 1 else np.random.rand() * 0.1
                ),
                "next.done": frame_idx == episode_length - 1,
                "next.success": frame_idx == episode_length - 1,
                # 全局索引
                "index": total_frames,
                # 任务索引 (默认单任务)
                "task_index": 0,
            }
            all_frames.append(frame_data)
            total_frames += 1

        if (ep_idx + 1) % 10 == 0:
            print(f"  Generated {ep_idx + 1}/{num_episodes} episodes...")

    print(f"✓ Generated {num_episodes} episodes ({total_frames} total frames)")

    # 转换为DataFrame
    df = pd.DataFrame(all_frames)

    # 保存为chunk (v2格式所有数据在一个chunk中，如果数据量大可以分多个chunk)
    num_chunks = (total_frames + chunks_size - 1) // chunks_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunks_size
        end_idx = min((chunk_idx + 1) * chunks_size, total_frames)

        chunk_df = df.iloc[start_idx:end_idx]

        chunk_dir = output_path / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = chunk_dir / "file-000.parquet"
        chunk_df.to_parquet(parquet_path, index=False)

        print(f"✓ Saved chunk {chunk_idx}: {len(chunk_df)} frames")

    # 生成stats.json (v2格式)
    print("Generating normalization stats...")

    # 计算统计信息
    action_data = np.array(df["action"].tolist())
    state_data = np.array(df["observation.state"].tolist())

    stats = {
        "action": {
            "mean": action_data.mean(axis=0).tolist(),
            "std": action_data.std(axis=0).tolist(),
            "min": action_data.min(axis=0).tolist(),
            "max": action_data.max(axis=0).tolist(),
            "q01": np.percentile(action_data, 1, axis=0).tolist(),
            "q99": np.percentile(action_data, 99, axis=0).tolist(),
        },
        "observation.state": {
            "mean": state_data.mean(axis=0).tolist(),
            "std": state_data.std(axis=0).tolist(),
            "min": state_data.min(axis=0).tolist(),
            "max": state_data.max(axis=0).tolist(),
            "q01": np.percentile(state_data, 1, axis=0).tolist(),
            "q99": np.percentile(state_data, 99, axis=0).tolist(),
        },
        # 添加其他字段的统计 (简化版本)
        "next.reward": {
            "mean": [float(df["next.reward"].mean())],
            "std": [float(df["next.reward"].std())],
            "min": [float(df["next.reward"].min())],
            "max": [float(df["next.reward"].max())],
        },
    }

    stats_path = output_path / "meta/stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Generated {stats_path}")

    # 生成info.json (v2格式)
    print("Generating dataset metadata...")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "humanoid_dummy",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.image": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": "dummy",  # 没有真实视频
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [proprio_dim],
                "names": {"proprioception": [f"proprio_{i}" for i in range(proprio_dim)]},
                "fps": float(fps),
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": {"motors": [f"motor_{i}" for i in range(action_dim)]},
                "fps": float(fps),
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
                "fps": float(fps),
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
                "fps": float(fps),
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
                "fps": float(fps),
            },
            "next.reward": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
                "fps": float(fps),
            },
            "next.done": {"dtype": "bool", "shape": [1], "names": None, "fps": float(fps)},
            "next.success": {"dtype": "bool", "shape": [1], "names": None, "fps": float(fps)},
            "index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
            "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        },
    }

    info_path = output_path / "meta/info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"✓ Generated {info_path}")

    # 生成episode元数据 (可选，用于快速查找episode边界)
    episodes_info = []
    for ep_idx in range(num_episodes):
        ep_info = {
            "episode_index": ep_idx,
            "length": episode_length,
            "timestamp": float(episode_length / fps),
        }
        episodes_info.append(ep_info)

    episodes_path = output_path / "meta/episodes.json"
    with open(episodes_path, "w") as f:
        json.dump(episodes_info, f, indent=2)
    print(f"✓ Generated {episodes_path}")

    # 打印总结
    print("\n" + "=" * 60)
    print("✓ Dummy dataset (LeRobot v2) created successfully!")
    print("=" * 60)
    print(f"Location: {output_dir}")
    print(f"Episodes: {num_episodes}")
    print(f"Total frames: {total_frames}")
    print(f"Action dim: {action_dim}")
    print(f"Proprio dim: {proprio_dim}")
    print(f"Chunks: {num_chunks}")
    print("\nDataset structure (v2 format):")
    print(f"  {output_dir}/")
    print(f"    ├── meta/")
    print(f"    │   ├── stats.json")
    print(f"    │   ├── info.json")
    print(f"    │   └── episodes.json")
    print(f"    ├── data/")
    print(f"    │   └── chunk-000/")
    print(f"    │       └── file-000.parquet")
    print(f"    └── videos/")
    print(f"        └── observation.image/")
    print(f"            └── chunk-000/")
    print("\nNext steps:")
    print("  1. Verify dataset:")
    print(f"     ls -lh {output_dir}/data/chunk-000/")
    print("  2. Test loading:")
    print("     python -c 'from RoboRenForce.dataset.lerobot import LeRobotDataset; ...'")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dummy LeRobot v2 dataset for testing"
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
    parser.add_argument(
        "--chunks_size",
        type=int,
        default=1000,
        help="Chunk size (frames per chunk)",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second (default: 10)"
    )

    args = parser.parse_args()

    generate_dummy_dataset_v2(
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
        chunks_size=args.chunks_size,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
