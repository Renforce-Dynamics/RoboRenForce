"""
Dataset Entry Registry

A DatasetEntry describes one dataset's metadata: where it is, what format,
which robot, and what transforms to apply. This is the base class —
concrete entries (e.g. Psi0G1PickApple) are defined in task packages.

The registry is NOT a global singleton; entries are composed into
MixtureDatasetCfg.entries as a list.
"""

from __future__ import annotations

from dataclasses import MISSING, field as dataclass_field

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template import ClassTemplateBaseCfg
from RoboRenForce.prototype.embodied_ai.robot import RobotCfg
from RoboRenForce.prototype.embodied_ai.transforms.base_transform import DataTransformCfg


@configclass
class DatasetEntryCfg(ClassTemplateBaseCfg):
    """
    Metadata for one dataset.

    Concrete entries set default values in RRF_vla_tasks. Example:

        @configclass
        class Psi0G1PickAppleCfg(DatasetEntryCfg):
            name = "psi0_g1_pick_apple"
            data_root = "/data/psi0/G1_Dex3_PickApple_lerobot"
            format = "lerobot_v2"
            robot_cfg = UnitreeG1Cfg()
            source = "psi0"
            tags = ["pick", "dexterous"]
    """

    # Identity
    name: str = MISSING

    # Data location & format
    data_root: str = MISSING
    format: str = "lerobot_v2"          # "lerobot_v2" | "hdf5" | ...

    # Associated robot
    robot_cfg: RobotCfg = RobotCfg()

    # Provenance
    source: str = ""                    # "psi0" / "aloha" / "bridge" / "custom"
    num_episodes: int = 0
    tags: list = []

    # Transforms applied to this dataset's samples (after loading, before mixing)
    # Typically: [RepackTransform, NormalizeTransform]
    transform_cfgs: list = []           # list[DataTransformCfg]

    # Dataset loader options
    load_videos: bool = False
    frames_dir: str = ""
    image_size: tuple = (224, 224)
