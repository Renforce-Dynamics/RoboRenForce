"""
Embodied Prototypes

Abstract definitions for embodied AI training (pretrain, SFT, RL fine-tune).
Concrete implementations live in task packages (source/tasks/RRF_*/).

Provides:
- EmbodiedEnv: VecEnv with multimodal obs (image, text, proprio)
- EmbodiedOutput: standardized env step / dataset sample output
- Robot / RobotCfg: action space abstraction
- DatasetEntryCfg: dataset metadata (source, format, robot)
- DataTransform: composable data transform pipeline
- MixtureDataset: multi-dataset mixing
- BasePolicy / ForwardType: unified policy interface for all training modes
"""

from .base_policy import BasePolicy, ForwardType
from .embodied_env import EmbodiedEnv
from .embodied_data import EmbodiedOutput, Trajectory
from .robot import Robot, RobotCfg
from .dataset_entry import DatasetEntryCfg
from .transforms import DataTransform, DataTransformCfg, ComposeTransform, ComposeTransformCfg
from .transforms import NormalizeTransform, NormalizeTransformCfg
from .mixture import MixtureDataset, MixtureDatasetCfg, BatchMixtureSampler
