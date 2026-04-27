"""Env cfg for D4RL Hopper.

Note: D4RL is state-only and does not match the multimodal VLA-RL contract.
This file is intentionally a placeholder so the registry/scripts contract
holds; ``build()`` raises a clear NotImplementedError until the
state-only-actor + offline→online finetune path lands.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class D4RLHopperEnvCfg:
    env_name: str = "hopper-medium-v2"
    num_envs: int = 1
    device: str = "cpu"
    action_dim: int = 3
    state_dim: int = 11
    max_episode_steps: int = 1000

    def build(self):
        raise NotImplementedError(
            "D4RL VLA-RL is not yet wired. D4RL is state-only; the VLA-RL "
            "pipeline currently expects multimodal envs. Track this in "
            "docs/PLAN-task2-vla-rl-task-packages.md (Step 4)."
        )
