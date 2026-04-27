"""PPO runner cfg for RoboTwin ``place_empty_cup``."""

from __future__ import annotations

from RoboRenForce.runners.vla.rl.vla_ppo_runner import VLAPPORunnerCfg
from RoboRenForce.utils.configclass import configclass

from .._shared.algo_presets import ppo_robotwin_default
from .actor import build_place_cup_policy


@configclass
class RoboTwinPlaceCupPPOCfg(VLAPPORunnerCfg):
    ppo_cfg = ppo_robotwin_default()

    log_interval: int = 1
    save_interval: int = 50
    checkpoint_dir: str = "checkpoints/RoboTwin-PlaceCup-PPO-v0"

    eval_interval: int = 50
    eval_episodes: int = 20

    build_policy = staticmethod(lambda: build_place_cup_policy(use_value_head=True))
