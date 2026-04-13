#!/usr/bin/env python3
"""
从LeRobot Hub (HuggingFace) 下载数据集

Usage:
    python scripts/data/download_lerobot_dataset.py \
        --repo_id lerobot/pusht \
        --output_dir data/lerobot/pusht

推荐的小数据集:
- lerobot/pusht (2D推箱子, ~2GB, 500 episodes)
- lerobot/aloha_sim_insertion_human (ALOHA仿真, ~1GB, 50 episodes)
- lerobot/xarm_lift_medium (真实机械臂, ~3GB, 200 episodes)
"""

import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")
    exit(1)


def download_dataset(repo_id: str, output_dir: str):
    """
    从HuggingFace Hub下载LeRobot数据集。

    Args:
        repo_id: HuggingFace repo ID (e.g., "lerobot/pusht")
        output_dir: 本地保存路径
    """
    print("=" * 60)
    print(f"Downloading LeRobot dataset: {repo_id}")
    print("=" * 60)
    print(f"Destination: {output_dir}")
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )

        print("\n" + "=" * 60)
        print("✓ Download complete!")
        print("=" * 60)

        # 验证数据集结构
        print("\nVerifying dataset structure...")

        found_components = []
        if (output_path / "meta").exists():
            found_components.append("meta/")
            if (output_path / "meta/stats.safetensors").exists():
                found_components.append("  stats.safetensors")
            if (output_path / "meta/info.json").exists():
                found_components.append("  info.json")

        if (output_path / "train").exists():
            found_components.append("train/")
            train_files = list((output_path / "train").glob("*.parquet"))
            found_components.append(f"  {len(train_files)} parquet files")

        if (output_path / "data").exists():
            found_components.append("data/ (LeRobot v2 format)")

        print("\nFound:")
        for comp in found_components:
            print(f"  ✓ {comp}")

        # 统计信息
        print("\nDataset contents:")
        for item in sorted(output_path.iterdir()):
            if item.is_dir():
                num_files = len(list(item.rglob("*")))
                print(f"  {item.name}/ ({num_files} files)")
            else:
                size_mb = item.stat().st_size / 1024 / 1024
                print(f"  {item.name} ({size_mb:.1f} MB)")

        print("\nNext steps:")
        print("  1. Verify dataset:")
        print(f"     ls -lh {output_dir}/")
        print("  2. Test loading:")
        print("     python -c 'from RoboRenForce.dataset.lerobot import LeRobotDataset; ...'")

    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("  - Check internet connection")
        print("  - Verify repo_id is correct")
        print("  - Try: huggingface-cli login (if dataset is private)")
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download LeRobot dataset from HuggingFace Hub"
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        default="lerobot/pusht",
        help="HuggingFace repo ID (e.g., lerobot/pusht)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/lerobot/pusht",
        help="Local directory to save dataset",
    )

    args = parser.parse_args()

    download_dataset(args.repo_id, args.output_dir)


if __name__ == "__main__":
    main()
