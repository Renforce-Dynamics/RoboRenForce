"""
DataTransform Base Class

All data transforms inherit from this. A transform takes a sample dict
and returns a modified sample dict.

Concrete transforms (robot-specific repack, VLM preprocess, etc.)
are implemented in task packages (source/tasks/RRF_*).
"""

from __future__ import annotations

from typing import Dict, Any

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template import ClassTemplateBase, ClassTemplateBaseCfg


class DataTransform(ClassTemplateBase):
    """Base class for all data transforms."""

    def __init__(self, cfg: DataTransformCfg):
        self.cfg = cfg

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample dict. Override in subclasses."""
        return sample


@configclass
class DataTransformCfg(ClassTemplateBaseCfg):
    """Base configuration for data transforms."""
    class_type: type[DataTransform] = DataTransform
