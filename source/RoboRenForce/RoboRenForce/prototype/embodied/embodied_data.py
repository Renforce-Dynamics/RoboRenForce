"""
EmbodiedOutput / Trajectory — Standardized data structures for embodied training.

Unifies the output format from:
- Sim environment step() → EmbodiedOutput
- Offline dataset sample  → EmbodiedOutput

So that runners and algorithms can be agnostic to the data source.

Reference: RLinf's rlinf/data/embodied_io_struct.py (EnvOutput / Trajectory)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class EmbodiedOutput:
    """Standardized output from one env step or one dataset sample.

    Both live sim and offline data produce this same structure.
    Runners consume this without caring where the data came from.

    Observation keys follow the EmbodiedEnv convention:
        main_images:       [B, H, W, C]
        wrist_images:      [B, H, W, C]  (optional)
        states:            [B, state_dim]
        task_descriptions: list[str]
    """

    obs: dict[str, Any]
    actions: torch.Tensor                          # [B, action_dim]
    rewards: Optional[torch.Tensor] = None         # [B]  (None for pretrain)
    dones: Optional[torch.Tensor] = None           # [B]
    terminations: Optional[torch.Tensor] = None    # [B]
    truncations: Optional[torch.Tensor] = None     # [B]

    # For dataset samples: next obs after action (if available)
    next_obs: Optional[dict[str, Any]] = None

    # Extra info (episode metrics, etc.)
    info: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def prepare_observations(raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Normalize raw observation dict to standard keys.

        Handles missing optional fields by setting them to None.
        """
        return {
            "main_images": raw_obs.get("main_images"),
            "wrist_images": raw_obs.get("wrist_images"),
            "states": raw_obs.get("states"),
            "task_descriptions": raw_obs.get("task_descriptions", []),
        }


@dataclass
class Trajectory:
    """A sequence of EmbodiedOutputs forming a rollout or episode chunk.

    Used by RL runners to store collected experience before algorithm update.
    """

    obs_seq: list[dict[str, Any]]       # T dicts, each with standard keys
    actions: torch.Tensor               # [T, B, action_dim]
    rewards: torch.Tensor               # [T, B]
    dones: torch.Tensor                 # [T, B]
    terminations: torch.Tensor          # [T, B]
    truncations: torch.Tensor           # [T, B]

    # Final observation (for bootstrapping value)
    final_obs: Optional[dict[str, Any]] = None

    info: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.actions.shape[0]

    @property
    def batch_size(self) -> int:
        return self.actions.shape[1]
