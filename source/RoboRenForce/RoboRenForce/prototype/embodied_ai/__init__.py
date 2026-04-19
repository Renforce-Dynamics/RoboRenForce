"""
Embodied AI Prototypes

Abstract definitions for VLA (Vision-Language-Action) and WAM (World-Action Model)
training paradigms. Concrete implementations live in task packages (RRF_vla_tasks).

Provides:
- Robot / RobotCfg: action space abstraction (open-loop, with closed-loop placeholder)
- DatasetEntryCfg: dataset metadata (source, format, robot association)
- DataTransform: composable data transform pipeline
- MixtureDataset: multi-dataset mixing (sampler / concat modes)
"""

from .robot import Robot, RobotCfg
from .dataset_entry import DatasetEntryCfg
from .transforms import DataTransform, DataTransformCfg, ComposeTransform, ComposeTransformCfg
from .transforms import NormalizeTransform, NormalizeTransformCfg
from .mixture import MixtureDataset, MixtureDatasetCfg, BatchMixtureSampler
