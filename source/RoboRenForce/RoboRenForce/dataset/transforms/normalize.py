"""
NormalizeTransform — stats-based normalization for action/state fields.

Loads mean/std from a stats file (JSON) and normalizes specified fields.
Supports per-dimension masking (zero-padded dims are skipped).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import field as dataclass_field

import torch
import numpy as np

from RoboRenForce.utils.configclass import configclass
from .base_transform import DataTransform, DataTransformCfg


class NormalizeTransform(DataTransform):
    """Stats-based normalization for action and/or state fields."""

    def __init__(self, cfg: NormalizeTransformCfg):
        super().__init__(cfg)
        self._stats = {}
        if cfg.stats_path:
            self._stats = self._load_stats(cfg.stats_path)

        self._action_mask = None
        if cfg.action_mask:
            self._action_mask = torch.tensor(cfg.action_mask, dtype=torch.bool)

    @staticmethod
    def _load_stats(stats_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        path = Path(stats_path)
        if not path.exists():
            return {}
        with open(path) as f:
            raw = json.load(f)
        return {k: {sk: np.array(sv) for sk, sv in v.items()} for k, v in raw.items()}

    def _normalize_field(self, data: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self._stats:
            return data
        stats = self._stats[key]
        mean = torch.tensor(stats["mean"], dtype=data.dtype, device=data.device)
        std = torch.tensor(stats["std"], dtype=data.dtype, device=data.device)
        result = (data - mean) / (std + 1e-8)

        # Apply mask: zero out padded dimensions
        if self._action_mask is not None and key == "action":
            mask = self._action_mask.to(device=data.device)
            result[..., ~mask] = 0.0

        return result

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.normalize_action and "action" in sample:
            sample["action"] = self._normalize_field(sample["action"], "action")

        state_key = "observation.state"
        if self.cfg.normalize_state and state_key in sample:
            sample[state_key] = self._normalize_field(sample[state_key], state_key)
            if "proprioception" in sample:
                sample["proprioception"] = sample[state_key]

        return sample

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Inverse transform for deployment."""
        if "action" not in self._stats:
            return action
        stats = self._stats["action"]
        mean = torch.tensor(stats["mean"], dtype=action.dtype, device=action.device)
        std = torch.tensor(stats["std"], dtype=action.dtype, device=action.device)
        return action * (std + 1e-8) + mean


@configclass
class NormalizeTransformCfg(DataTransformCfg):
    """Configuration for stats-based normalization."""
    class_type: type[NormalizeTransform] = NormalizeTransform
    stats_path: str = ""
    normalize_action: bool = True
    normalize_state: bool = True
    action_mask: list = []  # boolean mask: True = active dim
