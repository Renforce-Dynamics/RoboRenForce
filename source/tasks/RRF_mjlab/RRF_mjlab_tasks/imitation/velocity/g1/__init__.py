"""Register Unitree G1 velocity AMP task (MJLab backend) into RRF gym registry.

Mirrors the locomotion task registration convention; the env entry-point is
mjlab's ``ManagerBasedRlEnv`` plus an ``amp`` observation group attached by
:func:`unitree_g1_flat_amp_env_cfg` / :func:`unitree_g1_rough_amp_env_cfg`.

Launch via ``scripts/renforce/train_mjlab_amp.py`` which pre-wraps the env
with :class:`MJLabAMPEnvWrapper` (the AMP runner needs the seven-tuple step).
"""

import gymnasium as gym

from . import agents
from .amp_env_cfg import (
    unitree_g1_flat_amp_env_cfg,
    unitree_g1_rough_amp_env_cfg,
)


def _amp_env_cfg_loader(task_name: str, play: bool = False):
    """Resolve the AMP env cfg for a given RRF task name."""
    if "Flat" in task_name:
        return unitree_g1_flat_amp_env_cfg(play=play)
    return unitree_g1_rough_amp_env_cfg(play=play)


_TASKS = {
    "RoboRenForce-MJLab-AMP-Velocity-Flat-G1": ("Flat", unitree_g1_flat_amp_env_cfg),
    "RoboRenForce-MJLab-AMP-Velocity-Rough-G1": ("Rough", unitree_g1_rough_amp_env_cfg),
}


for task_id, (variant, _factory) in _TASKS.items():
    gym.register(
        id=task_id,
        entry_point="mjlab.envs:ManagerBasedRlEnv",
        disable_env_checker=True,
        kwargs={
            "RRF_amp_env_cfg_factory": (
                "RRF_mjlab_tasks.imitation.velocity.g1.amp_env_cfg:"
                f"unitree_g1_{variant.lower()}_amp_env_cfg"
            ),
            "RoboRenForce_entry_point": agents.MJLabG1AMPRunnerCfg().replace(
                experiment_name=task_id.replace("RoboRenForce-", "")
            ),
        },
    )

print(f"[INFO] Registered {len(_TASKS)} MJLab AMP velocity tasks for RoboRenForce.")
