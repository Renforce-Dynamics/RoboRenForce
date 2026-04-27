"""Placeholder SAC runner cfg for D4RL Hopper.

Subclasses VLAGRPORunnerCfg purely so the registry contract is satisfied;
the actual D4RL training path needs a state-only SAC runner that hasn't
been wired into the VLA-RL package yet.
"""

from __future__ import annotations

from RoboRenForce.runners.vla.rl.vla_grpo_runner import VLAGRPORunnerCfg
from RoboRenForce.utils.configclass import configclass


def _build_policy():
    raise NotImplementedError(
        "D4RL state-only policy not wired into VLA-RL package yet."
    )


@configclass
class D4RLHopperSACCfg(VLAGRPORunnerCfg):
    log_interval: int = 1
    save_interval: int = 50
    checkpoint_dir: str = "checkpoints/D4RL-Hopper-SAC-v0"
    eval_interval: int = 50
    eval_episodes: int = 10

    build_policy = staticmethod(_build_policy)
