"""D4RL VLA-RL task registrations (placeholder)."""

from __future__ import annotations

from ._registry import register_d4rl_task

from .hopper.env_cfg    import D4RLHopperEnvCfg
from .hopper.agents_sac import D4RLHopperSACCfg

register_d4rl_task(
    "D4RL-Hopper-SAC-v0",
    D4RLHopperEnvCfg(),
    D4RLHopperSACCfg(),
)

print("[INFO] Registered 1 D4RL VLA-RL task (Hopper-SAC, placeholder).")
