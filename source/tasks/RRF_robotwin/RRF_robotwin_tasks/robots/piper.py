"""
Piper Robot Definition (dual-arm)

The default embodiment for RoboTwin benchmark.
Bimanual setup: two Piper arms, each with 6 DoF + 1 gripper = 7 per arm.
Total action dim = 14.

Config in RoboTwin: embodiment = ["piper", "piper", 0.6]
"""

from __future__ import annotations

from RoboRenForce.prototype.embodied.robot import Robot, RobotCfg
from RoboRenForce.utils.configclass import configclass


PIPER_JOINT_NAMES = [
    # Left arm
    "left_joint1", "left_joint2", "left_joint3",
    "left_joint4", "left_joint5", "left_joint6",
    "left_gripper",
    # Right arm
    "right_joint1", "right_joint2", "right_joint3",
    "right_joint4", "right_joint5", "right_joint6",
    "right_gripper",
]


class Piper(Robot):
    """Dual Piper arms for RoboTwin tasks."""

    def repack_action(self, raw_action):
        """Pass through — Piper actions are already in the correct format."""
        return raw_action


@configclass
class PiperCfg(RobotCfg):
    """Piper bimanual robot configuration."""
    class_type: type[Piper] = Piper
    action_dim: int = 14         # 7 per arm (6 joints + 1 gripper)
    raw_action_dim: int = 14
    proprio_dim: int = 14        # joint positions (qpos)
    joint_names: list = PIPER_JOINT_NAMES
    joint_mapping: list = list(range(14))
