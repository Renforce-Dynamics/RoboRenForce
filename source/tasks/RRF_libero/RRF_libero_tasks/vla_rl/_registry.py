"""Idempotent gym task registration helper for LIBERO VLA-RL tasks."""

from __future__ import annotations

import gymnasium as gym


def _dummy_entry(*args, **kwargs):
    raise RuntimeError(
        "RRF VLA-RL tasks are not constructed via gym.make(). "
        "Use scripts/vla/rl/train_libero.py."
    )


def register_libero_task(task_id: str, env_cfg, runner_cfg) -> None:
    gym.envs.registry.pop(task_id, None)
    gym.register(
        id=task_id,
        entry_point="RRF_libero_tasks.vla_rl._registry:_dummy_entry",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "RoboRenForce_entry_point": runner_cfg,
        },
    )
