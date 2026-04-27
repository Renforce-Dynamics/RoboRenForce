"""
Psi0 G1 Dex3 PickApple Dataset Entry

Source:    https://github.com/physical-superintelligence-lab/Psi0
Hub:       https://huggingface.co/datasets/USC-PSI-Lab/psi-data
Download:  python scripts/data/download_psi0_dataset.py --task <task> --split real|simple
Robot:     Unitree G1 Dex3
Format:    LeRobot v2 (per-episode parquet + mp4)
Specs:     201 episodes, ~152K frames, 28D raw action -> 36D standardized, 32D state
"""

from __future__ import annotations

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.prototype.embodied import DatasetEntryCfg
from RRF_humanoid_psi0_tasks.robots import UnitreeG1Cfg


@configclass
class Psi0G1PickAppleCfg(DatasetEntryCfg):
    """Psi0 G1 Dex3 PickApple dataset entry."""

    name: str = "psi0_g1_pick_apple"
    data_root: str = "/data/psi0/G1_Dex3_PickApple_lerobot"
    format: str = "lerobot_v2"
    robot_cfg: UnitreeG1Cfg = UnitreeG1Cfg()
    source: str = "psi0"
    num_episodes: int = 201
    tags: list = ["pick", "dexterous", "bimanual"]

    load_videos: bool = False
    frames_dir: str = ""
    image_size: tuple = (224, 224)
