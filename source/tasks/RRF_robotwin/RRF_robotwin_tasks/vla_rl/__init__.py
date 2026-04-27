"""RoboTwin VLA-RL task registrations.

Importing this package registers ``RoboTwin-PlaceCup-GRPO-v0`` into
``gym.envs.registry``. Used by ``scripts/vla/rl/train_robotwin.py``.
"""

from __future__ import annotations

from ._registry import register_robotwin_task

from .place_empty_cup.env_cfg     import RoboTwinPlaceCupEnvCfg
from .place_empty_cup.agents_grpo import RoboTwinPlaceCupGRPOCfg

register_robotwin_task(
    "RoboTwin-PlaceCup-GRPO-v0",
    RoboTwinPlaceCupEnvCfg(),
    RoboTwinPlaceCupGRPOCfg(),
)

print("[INFO] Registered 1 RoboTwin VLA-RL task (PlaceCup-GRPO).")
