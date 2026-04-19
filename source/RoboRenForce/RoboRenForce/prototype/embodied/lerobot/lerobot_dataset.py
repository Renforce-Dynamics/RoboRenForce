"""
LeRobot Dataset Loader (v2 Format)

Loads datasets in LeRobot v2 Parquet format (chunk-based).
Supports both standard LeRobot v2 and Psi0-converted formats.

Format variants handled:
- Standard: data/chunk-000/file-000.parquet  (single file per chunk)
- Psi0:     data/chunk-000/episode_000000.parquet  (per-episode files)

Column mapping (Psi0 → standard):
- "states" → "observation.state" / "proprioception"
- Missing "next.reward" / "next.success" → defaults to 0.0 / False
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import MISSING, field
import json
import threading

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


# Column name aliases: maps alternative names → canonical names
_COLUMN_ALIASES = {
    "states": "observation.state",
    "state": "observation.state",
    "obs.state": "observation.state",
}


class LeRobotDataset(Dataset):
    """
    PyTorch dataset for LeRobot v2 format.

    IO:
        __init__(cfg) -> dataset
        __getitem__(idx) -> {
            "observation.state": (proprio_dim,),
            "proprioception": (proprio_dim,),
            "action": (action_dim,),
            "image": (C, H, W) if load_videos else absent,
            "episode_index": int,
            "frame_index": int,
            "timestamp": float,
            "next.done": bool,
        }
    """

    def __init__(self, cfg: LeRobotDatasetCfg):
        super().__init__()
        self.cfg = cfg

        self.data_root = Path(cfg.data_root)
        self.split = cfg.split
        self.load_videos = cfg.load_videos
        self.cache_size = cfg.cache_size

        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")

        # Load metadata
        self.info = self._load_info()
        self.stats = self._load_stats()
        self.episodes_info = self._load_episodes_metadata()

        # Detect column mapping from info.json features
        self._col_map = self._build_column_map()

        # Load data chunks
        self.chunks = self._load_chunks()

        # Build frame index
        self.total_frames = self.info["total_frames"]

        # Chunk cache
        self._chunk_cache: Dict[int, pd.DataFrame] = {}

        # Image / Video loading state
        self.frames_dir = Path(cfg.frames_dir) if cfg.frames_dir else None
        self._video_path_template = self.info.get("video_path", "")
        self._video_containers: Dict[str, Any] = {}
        self._video_lock = threading.Lock()
        self._chunks_size = self.info.get("chunks_size", 1000)

        # Detect video camera key from features
        self._video_keys = []
        for feat_name, feat_info in self.info.get("features", {}).items():
            if feat_info.get("dtype") == "video":
                self._video_keys.append(feat_name)

        # Infer dimensions from features
        features = self.info.get("features", {})
        state_key = self._col_map.get("observation.state", "observation.state")
        if state_key in features:
            self.proprio_dim = features[state_key]["shape"][0]
        else:
            self.proprio_dim = 0
        if "action" in features:
            self.action_dim = features["action"]["shape"][0]
        else:
            self.action_dim = 0

        print(f"Loaded LeRobot dataset: {self.data_root}")
        print(f"  Episodes: {self.info['total_episodes']}, Frames: {self.total_frames}")
        print(f"  Action dim: {self.action_dim}, State dim: {self.proprio_dim}")
        print(f"  Chunks: {len(self.chunks)}")

    def _load_info(self) -> Dict[str, Any]:
        info_path = self.data_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found: {info_path}")

        with open(info_path) as f:
            info = json.load(f)

        for field_name in ["total_episodes", "total_frames", "fps", "features"]:
            if field_name not in info:
                raise ValueError(f"Missing required field in info.json: {field_name}")

        return info

    def _load_stats(self) -> Dict[str, Any]:
        stats_path = self.data_root / "meta" / "stats.json"
        if not stats_path.exists():
            return {}

        with open(stats_path) as f:
            return json.load(f)

    def _load_episodes_metadata(self) -> Optional[List[Dict]]:
        # Try JSONL first (Psi0 format), then JSON
        jsonl_path = self.data_root / "meta" / "episodes.jsonl"
        if jsonl_path.exists():
            episodes = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        episodes.append(json.loads(line))
            return episodes

        json_path = self.data_root / "meta" / "episodes.json"
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)

        return None

    def _build_column_map(self) -> Dict[str, str]:
        """Build mapping from canonical name → actual column name in parquet."""
        features = self.info.get("features", {})
        col_map = {}

        # Check if 'states' exists instead of 'observation.state'
        if "states" in features and "observation.state" not in features:
            col_map["observation.state"] = "states"
        elif "state" in features and "observation.state" not in features:
            col_map["observation.state"] = "state"

        return col_map

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Find all data chunks, supporting both file-*.parquet and episode_*.parquet."""
        data_dir = self.data_root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        chunk_dirs = sorted(data_dir.glob("chunk-*"))
        if not chunk_dirs:
            raise ValueError(f"No chunk directories found in {data_dir}")

        chunks = []
        current_frame = 0

        for chunk_dir in chunk_dirs:
            chunk_index = int(chunk_dir.name.split("-")[1])

            # Try standard format first (file-*.parquet)
            parquet_files = sorted(chunk_dir.glob("file-*.parquet"))

            if not parquet_files:
                # Try Psi0 format (episode_*.parquet)
                parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))

            if not parquet_files:
                continue

            # Count total frames across all parquet files in this chunk
            # For per-episode format, we concatenate them
            num_frames = 0
            file_paths = []
            for pf in parquet_files:
                pf_meta = pd.read_parquet(pf, columns=["frame_index"])
                file_paths.append(pf)
                num_frames += len(pf_meta)

            chunk_info = {
                "chunk_index": chunk_index,
                "paths": file_paths,
                "start_frame": current_frame,
                "end_frame": current_frame + num_frames,
                "num_frames": num_frames,
            }
            chunks.append(chunk_info)
            current_frame += num_frames

        return chunks

    def _get_chunk_for_frame(self, frame_idx: int) -> int:
        for chunk_info in self.chunks:
            if chunk_info["start_frame"] <= frame_idx < chunk_info["end_frame"]:
                return chunk_info["chunk_index"]
        raise IndexError(f"Frame index {frame_idx} out of range [0, {self.total_frames})")

    def _load_chunk(self, chunk_index: int) -> pd.DataFrame:
        if chunk_index in self._chunk_cache:
            return self._chunk_cache[chunk_index]

        chunk_info = None
        for c in self.chunks:
            if c["chunk_index"] == chunk_index:
                chunk_info = c
                break
        if chunk_info is None:
            raise ValueError(f"Chunk {chunk_index} not found")

        # Load and concatenate all parquet files in chunk
        dfs = [pd.read_parquet(p) for p in chunk_info["paths"]]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # Cache management
        if len(self._chunk_cache) >= self.cache_size:
            oldest_key = next(iter(self._chunk_cache))
            del self._chunk_cache[oldest_key]

        self._chunk_cache[chunk_index] = df
        return df

    def _read_column(self, row, canonical_name: str, default=None):
        """Read a column using the column map, falling back to canonical name."""
        actual_name = self._col_map.get(canonical_name, canonical_name)
        if actual_name in row.index:
            return row[actual_name]
        if canonical_name in row.index:
            return row[canonical_name]
        return default

    def _load_frame_from_dir(self, episode_index: int, frame_index: int, image_size: tuple) -> torch.Tensor:
        """Load a pre-extracted JPEG frame from frames_dir."""
        from torchvision.transforms.functional import resize
        from torchvision.io import read_image

        frame_path = self.frames_dir / f"episode_{episode_index:06d}" / f"{frame_index:06d}.jpg"
        if not frame_path.exists():
            # Try PNG fallback
            frame_path = frame_path.with_suffix(".png")
        if not frame_path.exists():
            return torch.zeros(3, *image_size)

        img = read_image(str(frame_path)).float() / 255.0  # (C, H, W) float32
        if img.shape[1:] != image_size:
            img = resize(img, list(image_size))
        return img

    def _get_video_path(self, episode_index: int) -> Path:
        """Get video file path for a given episode."""
        chunk_idx = episode_index // self._chunks_size
        # Try template from info.json
        if self._video_path_template:
            rel = self._video_path_template.format(
                episode_chunk=chunk_idx, episode_index=episode_index,
            )
            path = self.data_root / rel
            if path.exists():
                return path

        # Fallback: search common patterns
        for camera in self._video_keys:
            # Psi0 pattern: videos/chunk-000/egocentric/episode_000000.mp4
            camera_short = camera.split(".")[-1]  # "observation.images.egocentric" -> "egocentric"
            path = self.data_root / "videos" / f"chunk-{chunk_idx:03d}" / camera_short / f"episode_{episode_index:06d}.mp4"
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Video not found for episode {episode_index} in {self.data_root / 'videos'}"
        )

    def _load_video_frame(self, episode_index: int, frame_index: int, image_size: tuple = (224, 224)) -> torch.Tensor:
        """
        Load a single video frame using PyAV.

        Opens a fresh container per call for DataLoader worker safety.
        For production workloads, consider using pre-extracted frames.

        Returns:
            (C, H, W) float32 tensor normalized to [0, 1]
        """
        import av
        from torchvision.transforms.functional import resize

        video_path = self._get_video_path(episode_index)

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        # Seek to target frame by index
        # For frame-accurate seek, go to the frame directly
        fps = float(stream.average_rate)
        time_base = float(stream.time_base)
        target_pts = int(frame_index / fps / time_base)
        container.seek(max(0, target_pts - 1), stream=stream)

        img_tensor = None
        for frame in container.decode(video=0):
            current_idx = int(frame.pts * time_base * fps + 0.5)
            if current_idx >= frame_index:
                img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_tensor = resize(img_tensor, list(image_size))
                break

        container.close()

        if img_tensor is None:
            return torch.zeros(3, *image_size)
        return img_tensor

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(f"Index {idx} out of range [0, {self.total_frames})")

        chunk_index = self._get_chunk_for_frame(idx)
        chunk_df = self._load_chunk(chunk_index)

        chunk_info = self.chunks[chunk_index]
        local_idx = idx - chunk_info["start_frame"]
        row = chunk_df.iloc[local_idx]

        # Build sample with column mapping
        state = self._read_column(row, "observation.state")
        action = row["action"]

        sample = {
            "action": torch.tensor(action, dtype=torch.float32),
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
            "timestamp": float(row["timestamp"]),
            "next.done": bool(row.get("next.done", False)),
        }

        # State / proprioception
        if state is not None:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            sample["observation.state"] = state_tensor
            sample["proprioception"] = state_tensor

        # Optional columns
        if "next.reward" in row.index:
            sample["next.reward"] = float(row["next.reward"])
        if "next.success" in row.index:
            sample["next.success"] = bool(row["next.success"])
        if "index" in row.index:
            sample["index"] = int(row["index"])
        if "task_index" in row.index:
            sample["task_index"] = int(row["task_index"])

        # Load image frame (prefer pre-extracted frames > video decode)
        if self.load_videos:
            ep_idx = sample["episode_index"]
            fr_idx = sample["frame_index"]
            if self.frames_dir and self.frames_dir.exists():
                sample["image"] = self._load_frame_from_dir(
                    ep_idx, fr_idx, image_size=self.cfg.image_size,
                )
            elif self._video_keys:
                sample["image"] = self._load_video_frame(
                    ep_idx, fr_idx, image_size=self.cfg.image_size,
                )

        return sample

    def get_episode(self, episode_index: int) -> List[Dict[str, torch.Tensor]]:
        frames = []
        for chunk_info in self.chunks:
            chunk_df = self._load_chunk(chunk_info["chunk_index"])
            episode_mask = chunk_df["episode_index"] == episode_index
            episode_df = chunk_df[episode_mask]

            if len(episode_df) == 0:
                continue

            for local_idx in range(len(episode_df)):
                row = episode_df.iloc[local_idx]
                state = self._read_column(row, "observation.state")
                action = row["action"]

                sample = {
                    "action": torch.tensor(action, dtype=torch.float32),
                    "episode_index": int(row["episode_index"]),
                    "frame_index": int(row["frame_index"]),
                    "timestamp": float(row["timestamp"]),
                    "next.done": bool(row.get("next.done", False)),
                }
                if state is not None:
                    sample["observation.state"] = torch.tensor(state, dtype=torch.float32)
                    sample["proprioception"] = sample["observation.state"]
                frames.append(sample)

        if not frames:
            raise ValueError(f"Episode {episode_index} not found")
        return frames

    def get_stats(self, key: str) -> Dict[str, np.ndarray]:
        if key not in self.stats:
            raise KeyError(f"Stats not found for key: {key}")
        return {k: np.array(v) for k, v in self.stats[key].items()}

    def normalize(self, data: torch.Tensor, key: str) -> torch.Tensor:
        stats = self.get_stats(key)
        mean = torch.tensor(stats["mean"], dtype=data.dtype, device=data.device)
        std = torch.tensor(stats["std"], dtype=data.dtype, device=data.device)
        return (data - mean) / (std + 1e-8)

    def denormalize(self, data: torch.Tensor, key: str) -> torch.Tensor:
        stats = self.get_stats(key)
        mean = torch.tensor(stats["mean"], dtype=data.dtype, device=data.device)
        std = torch.tensor(stats["std"], dtype=data.dtype, device=data.device)
        return data * (std + 1e-8) + mean


@configclass
class LeRobotDatasetCfg(ModuleBaseCfg):
    """
    LeRobot v2 dataset configuration.

    Supports both standard and Psi0-converted formats:
        data_root/
            meta/
                info.json
                stats.json (optional)
                episodes.json or episodes.jsonl (optional)
            data/
                chunk-000/
                    file-000.parquet  OR  episode_000000.parquet, episode_000001.parquet, ...
            videos/ (optional)
    """

    class_type: type[LeRobotDataset] = LeRobotDataset

    data_root: str = MISSING
    split: str = "train"

    load_videos: bool = False
    video_backend: str = "pyav"
    frames_dir: str = ""  # Pre-extracted frames directory (faster than video decode)
    image_size: tuple = (224, 224)
    cache_size: int = 100
