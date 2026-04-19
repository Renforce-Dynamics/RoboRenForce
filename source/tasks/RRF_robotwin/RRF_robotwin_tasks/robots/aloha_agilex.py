"""
ALOHA-Agilex Robot Definition (dual-arm)

Alternative embodiment for RoboTwin, used by pi0/pi0.5 models.
Config in RoboTwin: embodiment = ["aloha-agilex"]
"""

from __future__ import annotations

from RoboRenForce.prototype.embodied.robot import Robot, RobotCfg
from RoboRenForce.utils.configclass import configclass


class AlohaAgilex(Robot):
    """ALOHA-Agilex dual-arm robot for RoboTwin tasks."""

    def repack_action(self, raw_action):
        return raw_action


@configclass
class AlohaAgilexCfg(RobotCfg):
    """ALOHA-Agilex configuration."""
    class_type: type[AlohaAgilex] = AlohaAgilex
    action_dim: int = 14
    raw_action_dim: int = 14
    proprio_dim: int = 14
    joint_names: list = []
    joint_mapping: list = []
