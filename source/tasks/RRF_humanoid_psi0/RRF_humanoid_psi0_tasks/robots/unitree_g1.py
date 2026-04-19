"""
Unitree G1 Dex3 Robot Definition

Action space:
- Raw (from Psi0 data): 28D
  [left_arm(7) + left_hand(6) + right_arm(7) + right_hand(6) + waist(2)]
- Standardized: 36D
  Reordered into a canonical layout with zero-padding for unused dims.

State space: 32D proprioception

Reference: Psi0 HEPretrainRepackTransform
"""

from __future__ import annotations

import torch

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.prototype.embodied import Robot, RobotCfg

# G1 Dex3 joint mapping: raw 28D → standardized 36D
# This maps each raw dimension to its position in the 36D canonical space.
# Based on Psi0's joint reordering convention.
G1_DEX3_JOINT_MAP = [
    # waist (raw idx 26-27 → std idx 0-1)
    0, 1,
    # left_arm (raw idx 0-6 → std idx 2-8)
    2, 3, 4, 5, 6, 7, 8,
    # left_hand (raw idx 7-12 → std idx 9-14)
    9, 10, 11, 12, 13, 14,
    # right_arm (raw idx 13-19 → std idx 15-21)
    15, 16, 17, 18, 19, 20, 21,
    # right_hand (raw idx 20-25 → std idx 22-27)
    22, 23, 24, 25, 26, 27,
]

G1_DEX3_JOINT_NAMES = [
    "waist_yaw", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow_pitch", "left_elbow_roll", "left_wrist_yaw", "left_wrist_roll",
    "left_thumb", "left_index", "left_middle", "left_ring", "left_pinky", "left_thumb_rot",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow_pitch", "right_elbow_roll", "right_wrist_yaw", "right_wrist_roll",
    "right_thumb", "right_index", "right_middle", "right_ring", "right_pinky", "right_thumb_rot",
]


class UnitreeG1(Robot):
    """Unitree G1 Dex3 humanoid robot."""

    def repack_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """G1 Dex3 specific repack: 28D raw → 36D standard.

        Layout: [waist(2), left_arm(7), left_hand(6), right_arm(7), right_hand(6), padding(8)]
        Raw order: [left_arm(7), left_hand(6), right_arm(7), right_hand(6), waist(2)]
        """
        out = torch.zeros(*raw_action.shape[:-1], self.cfg.action_dim,
                          dtype=raw_action.dtype, device=raw_action.device)
        raw_dim = raw_action.shape[-1]

        if raw_dim == 28:
            # Reorder: waist first, then arms/hands
            out[..., 0:2] = raw_action[..., 26:28]    # waist
            out[..., 2:9] = raw_action[..., 0:7]      # left_arm
            out[..., 9:15] = raw_action[..., 7:13]    # left_hand
            out[..., 15:22] = raw_action[..., 13:20]  # right_arm
            out[..., 22:28] = raw_action[..., 20:26]  # right_hand
            # dims 28-35 remain zero (padding)
        elif raw_dim == self.cfg.action_dim:
            out = raw_action
        else:
            # Unknown raw dim, pass through with padding
            n = min(raw_dim, self.cfg.action_dim)
            out[..., :n] = raw_action[..., :n]

        return out


@configclass
class UnitreeG1Cfg(RobotCfg):
    """Unitree G1 Dex3 configuration."""

    class_type: type[UnitreeG1] = UnitreeG1

    name: str = "unitree_g1_dex3"
    action_dim: int = 36
    raw_action_dim: int = 28
    proprio_dim: int = 32
    joint_mapping: list = G1_DEX3_JOINT_MAP
    joint_names: list = G1_DEX3_JOINT_NAMES
