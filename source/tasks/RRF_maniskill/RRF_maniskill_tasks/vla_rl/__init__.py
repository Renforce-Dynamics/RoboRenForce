"""ManiSkill VLA-RL task registrations."""

from __future__ import annotations

from ._registry import register_maniskill_task

from .pick_cube.env_cfg     import ManiSkillPickCubeEnvCfg
from .pick_cube.agents_grpo import ManiSkillPickCubeGRPOCfg

register_maniskill_task(
    "ManiSkill-PickCube-GRPO-v0",
    ManiSkillPickCubeEnvCfg(),
    ManiSkillPickCubeGRPOCfg(),
)

print("[INFO] Registered 1 ManiSkill VLA-RL task (PickCube-GRPO).")
