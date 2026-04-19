"""
Unitree H1 Robot Definition

Action space: 42D (full-body humanoid)
State space: varies by configuration

Placeholder — fill in actual joint mapping when H1 data is available.
"""

from __future__ import annotations

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.prototype.embodied import Robot, RobotCfg


class UnitreeH1(Robot):
    """Unitree H1 humanoid robot."""
    pass


@configclass
class UnitreeH1Cfg(RobotCfg):
    """Unitree H1 configuration."""

    class_type: type[UnitreeH1] = UnitreeH1

    name: str = "unitree_h1"
    action_dim: int = 42
    raw_action_dim: int = 42
    proprio_dim: int = 0               # TBD when data is available
    joint_mapping: list = []            # Identity mapping (no repack needed)
    joint_names: list = []
