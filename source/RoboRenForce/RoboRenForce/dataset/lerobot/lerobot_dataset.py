"""
LeRobot Dataset Loader (v2 Format)

Loads datasets in LeRobot v2 Parquet format (chunk-based).

Reference: .references/lerobot/lerobot/common/datasets/lerobot_dataset.py

Adaptations:
- Uses RoboRenForce configclass instead of LeRobot's config system
- Supports LeRobot v2 chunk-based format
- Integrates with RoboRenForce ModuleBase architecture
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import MISSING
import json

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


class LeRobotDataset(Dataset):
    """
    PyTorch dataset for LeRobot v2 format.

    Features:
    - Chunk-based Parquet storage (v2 format)
    - Lazy frame loading
    - Normalization via stats.json
    - Multi-modal observations (state, images, actions)

    Current limitations:
    - Video loading not yet implemented (returns dummy images)
    - Only train split supported
    - No data augmentation yet
    """

    def __init__(self, cfg: "LeRobotDatasetCfg"):
        super().__init__()
        self.cfg = cfg

        self.data_root = Path(cfg.data_root)
        self.split = cfg.split
        self.load_videos = cfg.load_videos
        self.cache_size = cfg.cache_size

        # Validate dataset root exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")

        # Load metadata
        self.info = self._load_info()
        self.stats = self._load_stats()

        # Load episodes metadata (if available)
        self.episodes_info = self._load_episodes_metadata()

        # Load data chunks
        self.chunks = self._load_chunks()

        # Build frame index
        self.total_frames = self.info["total_frames"]

        # Simple LRU cache for chunks (load on demand)
        self._chunk_cache: Dict[int, pd.DataFrame] = {}

        print(f"Loaded LeRobot dataset: {self.data_root}")
        print(f"  Total episodes: {self.info['total_episodes']}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Chunks: {len(self.chunks)}")

    def _load_info(self) -> Dict[str, Any]:
        """Load info.json metadata."""
        info_path = self.data_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found: {info_path}")

        with open(info_path) as f:
            info = json.load(f)

        # Validate required fields
        required_fields = ["total_episodes", "total_frames", "fps", "features"]
        for field in required_fields:
            if field not in info:
                raise ValueError(f"Missing required field in info.json: {field}")

        return info

    def _load_stats(self) -> Dict[str, Any]:
        """Load stats.json normalization statistics."""
        stats_path = self.data_root / "meta" / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"stats.json not found: {stats_path}")

        with open(stats_path) as f:
            stats = json.load(f)

        return stats

    def _load_episodes_metadata(self) -> Optional[List[Dict]]:
        """Load episodes.json metadata (optional)."""
        episodes_path = self.data_root / "meta" / "episodes.json"
        if episodes_path.exists():
            with open(episodes_path) as f:
                return json.load(f)
        return None

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """
        Find all data chunks.

        Returns:
            List of chunk info dicts:
            [
                {
                    "chunk_index": 0,
                    "path": Path("data/chunk-000/file-000.parquet"),
                    "start_frame": 0,
                    "end_frame": 1000,
                },
                ...
            ]
        """
        data_dir = self.data_root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        chunks = []
        chunk_dirs = sorted(data_dir.glob("chunk-*"))

        if not chunk_dirs:
            raise ValueError(f"No chunk directories found in {data_dir}")

        current_frame = 0
        for chunk_dir in chunk_dirs:
            # Extract chunk index from directory name (e.g., "chunk-000" -> 0)
            chunk_index = int(chunk_dir.name.split("-")[1])

            # Find parquet file (usually file-000.parquet)
            parquet_files = list(chunk_dir.glob("file-*.parquet"))
            if not parquet_files:
                continue

            parquet_path = parquet_files[0]  # Use first file

            # Load parquet to get frame count (without loading full data)
            df = pd.read_parquet(parquet_path)
            num_frames = len(df)

            chunk_info = {
                "chunk_index": chunk_index,
                "path": parquet_path,
                "start_frame": current_frame,
                "end_frame": current_frame + num_frames,
                "num_frames": num_frames,
            }
            chunks.append(chunk_info)

            current_frame += num_frames

        return chunks

    def _get_chunk_for_frame(self, frame_idx: int) -> int:
        """
        Find which chunk contains the given frame index.

        Args:
            frame_idx: Global frame index

        Returns:
            chunk_index
        """
        for chunk_info in self.chunks:
            if chunk_info["start_frame"] <= frame_idx < chunk_info["end_frame"]:
                return chunk_info["chunk_index"]

        raise IndexError(f"Frame index {frame_idx} out of range [0, {self.total_frames})")

    def _load_chunk(self, chunk_index: int) -> pd.DataFrame:
        """
        Load chunk data (with caching).

        Args:
            chunk_index: Chunk index

        Returns:
            DataFrame with all frames in chunk
        """
        # Check cache first
        if chunk_index in self._chunk_cache:
            return self._chunk_cache[chunk_index]

        # Find chunk info
        chunk_info = None
        for c in self.chunks:
            if c["chunk_index"] == chunk_index:
                chunk_info = c
                break

        if chunk_info is None:
            raise ValueError(f"Chunk {chunk_index} not found")

        # Load parquet
        df = pd.read_parquet(chunk_info["path"])

        # Cache management (simple LRU: remove oldest if cache full)
        if len(self._chunk_cache) >= self.cache_size:
            # Remove first (oldest) item
            oldest_key = next(iter(self._chunk_cache))
            del self._chunk_cache[oldest_key]

        self._chunk_cache[chunk_index] = df

        return df

    def __len__(self) -> int:
        """Total number of frames in dataset."""
        return self.total_frames

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single frame.

        Args:
            idx: Global frame index

        Returns:
            {
                "observation.state": (proprio_dim,) float32 tensor,
                "action": (action_dim,) float32 tensor,
                "episode_index": int,
                "frame_index": int,
                "timestamp": float,
                "next.reward": float,
                "next.done": bool,
                "next.success": bool,
                # "observation.image": (C, H, W) tensor (TODO: Phase 1.2)
            }
        """
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(f"Index {idx} out of range [0, {self.total_frames})")

        # Find chunk
        chunk_index = self._get_chunk_for_frame(idx)
        chunk_df = self._load_chunk(chunk_index)

        # Get local index within chunk
        chunk_info = self.chunks[chunk_index]
        local_idx = idx - chunk_info["start_frame"]

        # Get row
        row = chunk_df.iloc[local_idx]

        # Convert to tensors
        sample = {
            "observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
            "action": torch.tensor(row["action"], dtype=torch.float32),
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
            "timestamp": float(row["timestamp"]),
            "next.reward": float(row["next.reward"]),
            "next.done": bool(row["next.done"]),
            "next.success": bool(row["next.success"]),
            "index": int(row["index"]),
            "task_index": int(row["task_index"]),
        }

        # TODO Phase 1.2: Load video frame if load_videos=True
        # if self.load_videos:
        #     sample["observation.image"] = self._load_image_frame(...)

        return sample

    def get_episode(self, episode_index: int) -> List[Dict[str, torch.Tensor]]:
        """
        Get all frames for a specific episode.

        Args:
            episode_index: Episode ID

        Returns:
            List of frame dicts
        """
        # Find all frames with matching episode_index
        frames = []

        for chunk_info in self.chunks:
            chunk_df = self._load_chunk(chunk_info["chunk_index"])

            # Filter by episode_index
            episode_mask = chunk_df["episode_index"] == episode_index
            episode_df = chunk_df[episode_mask]

            if len(episode_df) == 0:
                continue

            # Convert each row to sample dict
            for _, row in episode_df.iterrows():
                sample = {
                    "observation.state": torch.tensor(
                        row["observation.state"], dtype=torch.float32
                    ),
                    "action": torch.tensor(row["action"], dtype=torch.float32),
                    "episode_index": int(row["episode_index"]),
                    "frame_index": int(row["frame_index"]),
                    "timestamp": float(row["timestamp"]),
                    "next.reward": float(row["next.reward"]),
                    "next.done": bool(row["next.done"]),
                    "next.success": bool(row["next.success"]),
                }
                frames.append(sample)

        if not frames:
            raise ValueError(f"Episode {episode_index} not found")

        return frames

    def get_stats(self, key: str) -> Dict[str, np.ndarray]:
        """
        Get normalization statistics for a feature.

        Args:
            key: Feature name (e.g., "action", "observation.state")

        Returns:
            Dict with "mean", "std", "min", "max", "q01", "q99"
        """
        if key not in self.stats:
            raise KeyError(f"Stats not found for key: {key}")

        return {k: np.array(v) for k, v in self.stats[key].items()}

    def normalize(self, data: torch.Tensor, key: str) -> torch.Tensor:
        """
        Normalize data using stats.

        Args:
            data: Tensor to normalize
            key: Feature name

        Returns:
            Normalized tensor
        """
        stats = self.get_stats(key)
        mean = torch.tensor(stats["mean"], dtype=data.dtype, device=data.device)
        std = torch.tensor(stats["std"], dtype=data.dtype, device=data.device)

        return (data - mean) / (std + 1e-8)

    def denormalize(self, data: torch.Tensor, key: str) -> torch.Tensor:
        """
        Denormalize data using stats.

        Args:
            data: Normalized tensor
            key: Feature name

        Returns:
            Denormalized tensor
        """
        stats = self.get_stats(key)
        mean = torch.tensor(stats["mean"], dtype=data.dtype, device=data.device)
        std = torch.tensor(stats["std"], dtype=data.dtype, device=data.device)

        return data * (std + 1e-8) + mean


@configclass
class LeRobotDatasetCfg(ModuleBaseCfg):
    """
    LeRobot v2 dataset configuration.

    Dataset structure (v2):
        data_root/
            meta/
                info.json          # Dataset metadata + feature definitions
                stats.json         # Normalization statistics
                episodes.json      # (optional) Episode boundaries
            data/
                chunk-000/
                    file-000.parquet  # All episodes in chunks
                chunk-001/
                    ...
            videos/
                observation.image/
                    chunk-000/
                        file-000.mp4
    """

    class_type: type[LeRobotDataset] = LeRobotDataset

    # Data paths
    data_root: str = MISSING
    split: str = "train"  # Currently only "train" is supported (splits defined in info.json)

    # Processing
    # processor_cfg: LeRobotProcessorCfg = LeRobotProcessorCfg()  # TODO: Uncomment when processor is implemented

    # Performance
    load_videos: bool = False  # TODO: Implement video loading in Phase 1.2
    video_backend: str = "pyav"  # "pyav" or "opencv"
    cache_size: int = 100  # Number of frames to cache in memory
