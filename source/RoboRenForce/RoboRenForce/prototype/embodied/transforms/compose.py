"""
ComposeTransform — chain multiple transforms sequentially.
"""

from __future__ import annotations

from typing import Dict, Any
from dataclasses import field as dataclass_field

from RoboRenForce.utils.configclass import configclass
from .base_transform import DataTransform, DataTransformCfg


class ComposeTransform(DataTransform):
    """Apply a list of transforms in order."""

    def __init__(self, cfg: ComposeTransformCfg):
        super().__init__(cfg)
        self.transforms = [t_cfg.construct_from_cfg() for t_cfg in cfg.transform_cfgs]

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


@configclass
class ComposeTransformCfg(DataTransformCfg):
    """Configuration for chaining transforms."""
    class_type: type[ComposeTransform] = ComposeTransform
    transform_cfgs: list = []  # list[DataTransformCfg]
