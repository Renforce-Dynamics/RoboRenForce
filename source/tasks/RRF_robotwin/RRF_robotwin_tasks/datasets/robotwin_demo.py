"""
RoboTwin Demonstration Dataset Entry

Template for RoboTwin offline demonstration datasets.
Override task_name and data_root per task.
"""

from __future__ import annotations

from RoboRenForce.prototype.embodied.dataset_entry import DatasetEntryCfg
from RoboRenForce.utils.configclass import configclass
from RRF_robotwin_tasks.robots.piper import PiperCfg


@configclass
class RoboTwinDemoCfg(DatasetEntryCfg):
    """Base config for RoboTwin demonstration datasets."""
    name: str = "robotwin_demo"
    data_root: str = ""                    # path to dataset directory
    format: str = "lerobot_v2"
    robot_cfg: PiperCfg = PiperCfg()
    source: str = "robotwin"

    # Task metadata
    task_name: str = "place_empty_cup"     # matches RoboTwin task name
    num_episodes: int = 0
    tags: list = None
    load_videos: bool = False
    image_size: tuple = (224, 224)

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["bimanual", self.task_name]
