"""
Robot Base Definition

Describes a robot's action/state space independently of training paradigm (VLA/RL).

Two modes:
- Open-loop: action space only (for offline VLA pretraining from demos)
- Closed-loop: extends with simulation interface (for RL, deferred)

Concrete robot definitions (G1, H1, etc.) live in task packages (source/tasks/RRF_*) or
RRF_isaaclab_tasks, NOT here. This module only provides the base classes.
"""

from __future__ import annotations

from dataclasses import MISSING, field as dataclass_field

import torch

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template import ClassTemplateBase, ClassTemplateBaseCfg


class Robot(ClassTemplateBase):
    """
    Robot instance providing action-space operations.

    Subclass in task packages (source/tasks/RRF_*) / RRF_isaaclab_tasks to add robot-specific logic.
    """

    def __init__(self, cfg: RobotCfg):
        self.cfg = cfg

    # ------ open-loop interface (action space mapping) ------

    def repack_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Map raw collected action to standardized action space.

        Default: use joint_mapping index list to scatter raw dims into
        a zero-padded tensor of size (action_dim,).
        If no mapping is set, returns raw_action unchanged.
        """
        if not self.cfg.joint_mapping:
            return raw_action
        out = torch.zeros(*raw_action.shape[:-1], self.cfg.action_dim,
                          dtype=raw_action.dtype, device=raw_action.device)
        mapping = torch.tensor(self.cfg.joint_mapping, dtype=torch.long,
                               device=raw_action.device)
        out[..., mapping] = raw_action[..., :len(mapping)]
        return out

    def unrepack_action(self, action: torch.Tensor) -> torch.Tensor:
        """Inverse of repack: standardized → raw (for deployment)."""
        if not self.cfg.joint_mapping:
            return action
        mapping = torch.tensor(self.cfg.joint_mapping, dtype=torch.long,
                               device=action.device)
        return action[..., mapping]

    def get_action_mask(self) -> torch.Tensor:
        """Boolean mask: True for active dims, False for zero-padded dims."""
        mask = torch.zeros(self.cfg.action_dim, dtype=torch.bool)
        if self.cfg.joint_mapping:
            mapping = torch.tensor(self.cfg.joint_mapping, dtype=torch.long)
            mask[mapping] = True
        else:
            mask[:] = True
        return mask

    # ------ closed-loop interface (simulation, deferred) ------
    # These will be implemented when integrating with IsaacLab / MuJoCo.
    # Concrete closed-loop robots should override these in RRF_isaaclab_tasks.

    def reset(self, **kwargs):
        """Reset robot in simulation. Override in closed-loop subclass."""
        raise NotImplementedError("Closed-loop interface not implemented for this robot")

    def step(self, action: torch.Tensor):
        """Step robot in simulation. Override in closed-loop subclass."""
        raise NotImplementedError("Closed-loop interface not implemented for this robot")


@configclass
class RobotCfg(ClassTemplateBaseCfg):
    """
    Robot configuration base class.

    Describes the robot's action/observation space.
    Concrete configs (G1Cfg, H1Cfg, ...) are defined in task packages.
    """

    class_type: type[Robot] = Robot

    # Identity
    name: str = MISSING

    # Open-loop action space
    action_dim: int = MISSING           # Standardized action dimension
    raw_action_dim: int = 0             # Raw collected dimension (0 = same as action_dim)
    proprio_dim: int = 0                # Proprioception / state dimension

    # Joint mapping: indices mapping raw_action → standardized action
    # e.g. [0, 1, 5, 6, ...] means raw[0] → std[0], raw[1] → std[1], raw[2] → std[5], ...
    joint_mapping: list = []
    joint_names: list = []

    # Closed-loop (deferred, placeholders)
    # sim_engine: str = ""
    # urdf_path: str = ""
