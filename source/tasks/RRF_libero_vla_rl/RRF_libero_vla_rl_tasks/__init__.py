"""LIBERO VLA-RL task registrations.

Importing this package registers LIBERO-*-{GRPO}-v0 task IDs.
"""

from __future__ import annotations

from ._registry import register_libero_task

from .spatial_pick_object.env_cfg     import LiberoSpatialEnvCfg
from .spatial_pick_object.agents_grpo import LiberoSpatialGRPOCfg

register_libero_task(
    "LIBERO-Spatial-GRPO-v0",
    LiberoSpatialEnvCfg(),
    LiberoSpatialGRPOCfg(),
)

print("[INFO] Registered 1 LIBERO VLA-RL task (Spatial-GRPO).")
