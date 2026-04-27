"""RoboTwin VLA-RL task registrations.

Importing this package registers all RoboTwin-*-{GRPO,PPO}-v0 task IDs into
``gym.envs.registry``. Used by ``scripts/vla/rl/train_robotwin.py``.
"""

from __future__ import annotations

from ._registry import register_robotwin_task

# ---- place_empty_cup ----
from .place_empty_cup.env_cfg     import RoboTwinPlaceCupEnvCfg
from .place_empty_cup.agents_grpo import RoboTwinPlaceCupGRPOCfg
from .place_empty_cup.agents_ppo  import RoboTwinPlaceCupPPOCfg

register_robotwin_task(
    "RoboTwin-PlaceCup-GRPO-v0",
    RoboTwinPlaceCupEnvCfg(),
    RoboTwinPlaceCupGRPOCfg(),
)
register_robotwin_task(
    "RoboTwin-PlaceCup-PPO-v0",
    RoboTwinPlaceCupEnvCfg(),
    RoboTwinPlaceCupPPOCfg(),
)

print("[INFO] Registered 2 RoboTwin VLA-RL tasks (PlaceCup-GRPO, PlaceCup-PPO).")
