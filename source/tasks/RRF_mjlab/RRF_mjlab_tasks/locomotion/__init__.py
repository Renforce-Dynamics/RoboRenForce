"""MJLab locomotion tasks registered for RoboRenForce.

Registers Go1 (quadruped) and G1 (humanoid) velocity tracking tasks
with PPO and SAC agent configurations.

Entry point uses MJLab's ManagerBasedRlEnv, wrapped by our MJLab wrappers.
"""

import gymnasium as gym
from . import agents_ppo, agents_sac

# ============================================================================
# MJLab task IDs → (display_name, mjlab_task_id)
# These reference tasks registered in mjlab.tasks.velocity.config.*
# ============================================================================

_GO1_TASKS = {
    "Go1Flat": "Mjlab-Velocity-Flat-Unitree-Go1",
    "Go1Rough": "Mjlab-Velocity-Rough-Unitree-Go1",
}

_G1_TASKS = {
    "G1Flat": "Mjlab-Velocity-Flat-Unitree-G1",
    "G1Rough": "Mjlab-Velocity-Rough-Unitree-G1",
}


def _make_kwargs(mjlab_task_id: str, rrf_cfg):
    """Build gym.register kwargs for a MJLab-backed RRF task."""
    return {
        "mjlab_task_id": mjlab_task_id,
        "env_cfg_entry_point": f"RRF_mjlab_tasks.mjlab_utils.gym_config:load_mjlab_env_cfg",
        "RoboRenForce_entry_point": rrf_cfg,
    }


# ============================================================================
# Register Go1 tasks — PPO and SAC
# ============================================================================

for name, mjlab_id in _GO1_TASKS.items():
    # PPO
    gym.register(
        id=f"RoboRenForce-MJLab-{name}-PPO",
        entry_point="mjlab.envs:ManagerBasedRlEnv",
        disable_env_checker=True,
        kwargs=_make_kwargs(
            mjlab_id,
            agents_ppo.MJLabLocoPPOCfg().replace(experiment_name=f"MJLab-{name}-PPO"),
        ),
    )
    # SAC
    gym.register(
        id=f"RoboRenForce-MJLab-{name}-SAC",
        entry_point="mjlab.envs:ManagerBasedRlEnv",
        disable_env_checker=True,
        kwargs=_make_kwargs(
            mjlab_id,
            agents_sac.MJLabLocoSACCfg().replace(experiment_name=f"MJLab-{name}-SAC"),
        ),
    )

# ============================================================================
# Register G1 tasks — PPO and SAC
# ============================================================================

for name, mjlab_id in _G1_TASKS.items():
    # PPO (G1-specific config with longer training)
    gym.register(
        id=f"RoboRenForce-MJLab-{name}-PPO",
        entry_point="mjlab.envs:ManagerBasedRlEnv",
        disable_env_checker=True,
        kwargs=_make_kwargs(
            mjlab_id,
            agents_ppo.MJLabLocoG1PPOCfg().replace(experiment_name=f"MJLab-{name}-PPO"),
        ),
    )
    # SAC
    gym.register(
        id=f"RoboRenForce-MJLab-{name}-SAC",
        entry_point="mjlab.envs:ManagerBasedRlEnv",
        disable_env_checker=True,
        kwargs=_make_kwargs(
            mjlab_id,
            agents_sac.MJLabLocoSACCfg().replace(experiment_name=f"MJLab-{name}-SAC"),
        ),
    )

_total = len(_GO1_TASKS) * 2 + len(_G1_TASKS) * 2
print(f"[INFO] Registered {_total} MJLab locomotion tasks for RoboRenForce.")
