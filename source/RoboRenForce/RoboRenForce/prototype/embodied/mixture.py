"""
MixtureDataset — multi-dataset mixing with two modes.

Mode "sampler":
    Datasets remain separate. A custom BatchMixtureSampler draws from each
    dataset according to mix_weights. Each sample is transformed by its
    dataset-specific transforms then output in a unified format.

Mode "concat":
    All datasets are logically concatenated into one. Indexing is linear.
    Each sample goes through its own dataset's transforms before returning.

In both modes, the Robot associated with each entry handles action repack
(raw → standardized), and per-dataset transforms handle normalization.
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset, Sampler

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template import ClassTemplateBaseCfg
from RoboRenForce.prototype.embodied.robot import Robot
from RoboRenForce.prototype.embodied.dataset_entry import DatasetEntryCfg
from RoboRenForce.prototype.embodied.transforms.compose import ComposeTransform, ComposeTransformCfg


def _build_dataset_from_entry(entry: DatasetEntryCfg):
    """Instantiate a PyTorch Dataset from a DatasetEntryCfg based on format."""
    if entry.format == "lerobot_v2":
        from RoboRenForce.prototype.embodied.lerobot.lerobot_dataset import LeRobotDataset, LeRobotDatasetCfg
        ds_cfg = LeRobotDatasetCfg(
            data_root=entry.data_root,
            load_videos=entry.load_videos,
            frames_dir=entry.frames_dir,
            image_size=entry.image_size,
        )
        return LeRobotDataset(ds_cfg)
    else:
        raise ValueError(f"Unknown dataset format: {entry.format}")


def _build_transform(entry: DatasetEntryCfg) -> Optional[ComposeTransform]:
    """Build composed transform from entry's transform_cfgs."""
    if not entry.transform_cfgs:
        return None
    compose_cfg = ComposeTransformCfg(transform_cfgs=entry.transform_cfgs)
    return ComposeTransform(compose_cfg)


class MixtureDataset(Dataset):
    """
    Multi-dataset mixer.

    Each sub-dataset is associated with a Robot and a transform pipeline.
    Samples are repacked through Robot.repack_action() then transformed.
    """

    def __init__(self, cfg: MixtureDatasetCfg):
        super().__init__()
        self.cfg = cfg

        self.datasets: List[Dataset] = []
        self.robots: List[Robot] = []
        self.transforms: List[Optional[ComposeTransform]] = []
        self.entry_names: List[str] = []

        # Cumulative lengths for concat mode indexing
        self._cum_lengths: List[int] = []
        total = 0

        for entry_cfg in cfg.entries:
            ds = _build_dataset_from_entry(entry_cfg)
            robot = Robot(entry_cfg.robot_cfg) if entry_cfg.robot_cfg.name else None
            transform = _build_transform(entry_cfg)

            self.datasets.append(ds)
            self.robots.append(robot)
            self.transforms.append(transform)
            self.entry_names.append(entry_cfg.name)

            total += len(ds)
            self._cum_lengths.append(total)

        self._total_len = total

        # Determine unified action dim
        if cfg.unified_action_dim > 0:
            self.action_dim = cfg.unified_action_dim
        else:
            # Take max across all entries
            dims = [e.robot_cfg.action_dim for e in cfg.entries if e.robot_cfg.action_dim > 0]
            self.action_dim = max(dims) if dims else 0

        print(f"MixtureDataset: {len(self.datasets)} datasets, "
              f"{self._total_len} total samples, "
              f"unified_action_dim={self.action_dim}, "
              f"mode={cfg.mix_mode}")
        for i, name in enumerate(self.entry_names):
            print(f"  [{i}] {name}: {len(self.datasets[i])} samples")

    def _resolve_index(self, idx: int):
        """Map global index → (dataset_id, local_idx)."""
        for ds_id, cum in enumerate(self._cum_lengths):
            if idx < cum:
                local = idx - (self._cum_lengths[ds_id - 1] if ds_id > 0 else 0)
                return ds_id, local
        raise IndexError(f"Index {idx} out of range [0, {self._total_len})")

    def _process_sample(self, sample: Dict[str, Any], ds_id: int) -> Dict[str, Any]:
        """Apply robot repack + transforms to a sample."""
        # Robot action repack
        robot = self.robots[ds_id]
        if robot is not None and "action" in sample:
            sample["action"] = robot.repack_action(sample["action"])

        # Pad action to unified dim if needed
        if self.action_dim > 0 and "action" in sample:
            act = sample["action"]
            if act.shape[-1] < self.action_dim:
                pad = torch.zeros(*act.shape[:-1], self.action_dim - act.shape[-1],
                                  dtype=act.dtype, device=act.device)
                sample["action"] = torch.cat([act, pad], dim=-1)

        # Per-dataset transforms
        transform = self.transforms[ds_id]
        if transform is not None:
            sample = transform(sample)

        # Add dataset metadata
        sample["dataset_id"] = ds_id
        sample["dataset_name"] = self.entry_names[ds_id]

        return sample

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx) -> Dict[str, Any]:
        if isinstance(idx, tuple):
            # Tuple index: (dataset_id, local_idx) — used by BatchMixtureSampler
            ds_id, local_idx = idx
        else:
            # Linear index — concat mode
            ds_id, local_idx = self._resolve_index(idx)

        sample = self.datasets[ds_id][local_idx]
        return self._process_sample(sample, ds_id)

    def get_dataset_lengths(self) -> List[int]:
        """Get per-dataset lengths (for sampler construction)."""
        return [len(ds) for ds in self.datasets]


class BatchMixtureSampler(Sampler):
    """
    Weighted sampler for MixtureDataset (sampler mode).

    Each batch is drawn from a single dataset, selected by mix_weights.
    Yields tuples (dataset_id, local_idx) for MixtureDataset.__getitem__.
    """

    def __init__(
        self,
        dataset: MixtureDataset,
        batch_size: int,
        mix_weights: Optional[List[float]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        lengths = dataset.get_dataset_lengths()
        n = len(lengths)

        if mix_weights is None:
            # Proportional to dataset size
            total = sum(lengths)
            self.weights = [l / total for l in lengths]
        else:
            assert len(mix_weights) == n
            s = sum(mix_weights)
            self.weights = [w / s for w in mix_weights]

        # Pre-compute number of batches per dataset per epoch
        total_batches = sum(
            math.ceil(l / batch_size) if not drop_last else l // batch_size
            for l in lengths
        )
        self._total_batches = total_batches
        self._lengths = lengths

    def __iter__(self):
        # Build per-dataset index lists
        indices_per_ds = []
        for ds_id, length in enumerate(self._lengths):
            idxs = list(range(length))
            if self.shuffle:
                import random
                random.shuffle(idxs)
            indices_per_ds.append(idxs)

        # Interleave batches according to weights
        cursors = [0] * len(self._lengths)
        batches = []

        for ds_id, idxs in enumerate(indices_per_ds):
            c = 0
            while c < len(idxs):
                batch = [(ds_id, idxs[j]) for j in range(c, min(c + self.batch_size, len(idxs)))]
                if self.drop_last and len(batch) < self.batch_size:
                    break
                batches.append(batch)
                c += self.batch_size

        # Shuffle batch order (weighted interleave)
        if self.shuffle:
            import random
            random.shuffle(batches)

        for batch in batches:
            yield from batch

    def __len__(self):
        return sum(
            math.ceil(l / self.batch_size) * self.batch_size
            if not self.drop_last else (l // self.batch_size) * self.batch_size
            for l in self._lengths
        )


@configclass
class MixtureDatasetCfg(ClassTemplateBaseCfg):
    """
    Configuration for multi-dataset mixing.

    mix_mode:
        "sampler" — datasets stay separate, BatchMixtureSampler handles mixing
        "concat"  — datasets concatenated, standard DataLoader indexing
    """

    class_type: type[MixtureDataset] = MixtureDataset

    entries: list = []                  # list[DatasetEntryCfg]
    mix_mode: str = "concat"            # "sampler" | "concat"
    mix_weights: list = []              # weights for sampler mode (empty = proportional)
    unified_action_dim: int = 0         # 0 = auto-detect (max across entries)
