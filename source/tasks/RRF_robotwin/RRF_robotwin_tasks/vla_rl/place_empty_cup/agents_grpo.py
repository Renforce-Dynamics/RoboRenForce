"""GRPO runner cfg for RoboTwin ``place_empty_cup``."""

from __future__ import annotations

from RoboRenForce.runners.vla.rl.vla_grpo_runner import VLAGRPORunnerCfg
from RoboRenForce.utils.configclass import configclass

from .._shared.algo_presets import grpo_robotwin_default
from .actor import build_place_cup_policy


@configclass
class RoboTwinPlaceCupGRPOCfg(VLAGRPORunnerCfg):
    """Bundles GRPO algo cfg + (build_policy) callable + checkpoint paths.

    The script reads ``build_policy`` to construct the policy after applying
    any CLI overrides — the runner cfg itself is just the (algo, log) part.
    """

    grpo_cfg = grpo_robotwin_default()

    log_interval: int = 1
    save_interval: int = 50
    checkpoint_dir: str = "checkpoints/RoboTwin-PlaceCup-GRPO-v0"

    eval_interval: int = 50
    eval_episodes: int = 20

    # Used by the build helper. The script invokes ``build_policy()``
    # after any CLI overrides on the env cfg.
    build_policy = staticmethod(lambda: build_place_cup_policy(use_value_head=False))
