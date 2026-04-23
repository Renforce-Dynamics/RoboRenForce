"""Classic Gymnasium tasks registered for RoboRenForce.

Registers MuJoCo locomotion and classic control tasks with
PPO / SAC / DSAC / DSACT agent configurations.

Env IDs: RoboRenForce-Gym-{EnvName}-{Algorithm}
"""

import gymnasium as gym
from . import agents

# ============================================================================
# Task registry: short_name → gymnasium task ID
# ============================================================================

_MUJOCO_LOCOMOTION = {
    "HalfCheetah":  "HalfCheetah-v4",
    "Walker2d":     "Walker2d-v4",
    "Hopper":       "Hopper-v4",
    "Ant":          "Ant-v4",
    "Humanoid":     "Humanoid-v4",
    "Swimmer":      "Swimmer-v4",
}

_MUJOCO_MANIPULATION = {
    "Reacher":      "Reacher-v4",
    "Pusher":       "Pusher-v4",
    "InvertedPendulum":        "InvertedPendulum-v4",
    "InvertedDoublePendulum":  "InvertedDoublePendulum-v4",
}

_CLASSIC_CONTROL = {
    "Pendulum":     "Pendulum-v1",
}

# Merge all continuous-action tasks
_ALL_TASKS = {}
_ALL_TASKS.update(_MUJOCO_LOCOMOTION)
_ALL_TASKS.update(_MUJOCO_MANIPULATION)
_ALL_TASKS.update(_CLASSIC_CONTROL)

# Algorithm → runner config class
_ALGORITHMS = {
    "PPO":   agents.GymPPOCfg,
    "SAC":   agents.GymSACCfg,
    "DSAC":  agents.GymDSACCfg,
    "DSACT": agents.GymDSACTCfg,
}


def _make_kwargs(gym_task_id: str, rrf_cfg):
    return {
        "gym_task_id": gym_task_id,
        "RoboRenForce_entry_point": rrf_cfg,
    }


# ============================================================================
# Register all tasks × algorithms
# ============================================================================

_count = 0
for name, gym_id in _ALL_TASKS.items():
    for alg_name, alg_cfg_cls in _ALGORITHMS.items():
        env_id = f"RoboRenForce-Gym-{name}-{alg_name}"
        cfg = alg_cfg_cls()
        cfg.experiment_name = f"Gym-{name}-{alg_name}"
        gym.register(
            id=env_id,
            entry_point="RRF_gym_tasks.envs.gym_vec_env:GymVecEnv",
            disable_env_checker=True,
            kwargs=_make_kwargs(gym_id, cfg),
        )
        _count += 1

# Expose task/algorithm lists for programmatic use
TASK_NAMES = list(_ALL_TASKS.keys())
TASK_IDS = {
    f"RoboRenForce-Gym-{name}-{alg}": gym_id
    for name, gym_id in _ALL_TASKS.items()
    for alg in _ALGORITHMS
}
