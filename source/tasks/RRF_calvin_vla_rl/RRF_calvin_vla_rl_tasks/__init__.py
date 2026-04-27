"""CALVIN VLA-RL task registrations."""

from __future__ import annotations

from ._registry import register_calvin_task

from .d_split.env_cfg     import CalvinDSplitEnvCfg
from .d_split.agents_grpo import CalvinDSplitGRPOCfg

register_calvin_task(
    "CALVIN-D-GRPO-v0",
    CalvinDSplitEnvCfg(),
    CalvinDSplitGRPOCfg(),
)

print("[INFO] Registered 1 CALVIN VLA-RL task (D-GRPO).")
