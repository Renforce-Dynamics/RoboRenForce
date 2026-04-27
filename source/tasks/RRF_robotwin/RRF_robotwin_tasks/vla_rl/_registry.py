"""Idempotent gym task registration helper for RoboTwin VLA-RL tasks.

Each task ID resolves to:
  - ``env_cfg_entry_point``      : an EnvCfg instance (calling ``.build()`` produces the env)
  - ``RoboRenForce_entry_point`` : a Runner cfg instance (algo + policy + logger)

The script (:mod:`scripts.vla.rl._common`) reads both via ``gym.spec(task_id).kwargs``
and constructs the runner without ever importing benchmark-specific code.
"""

from __future__ import annotations

import gymnasium as gym


_ENTRY_POINT = "RRF_robotwin_tasks.vla_rl._registry:_dummy_entry"


def _dummy_entry(*args, **kwargs):
    """Placeholder ``gym.make`` entry point.

    The VLA RL training scripts never call ``gym.make`` — they read
    ``spec.kwargs["env_cfg_entry_point"]`` and build the env explicitly.
    Gym still requires *some* entry point at registration time.
    """
    raise RuntimeError(
        "RRF VLA-RL tasks are not constructed via gym.make(). "
        "Use scripts/vla/rl/train_<benchmark>.py which reads env_cfg_entry_point."
    )


def register_robotwin_task(task_id: str, env_cfg, runner_cfg) -> None:
    """Register (or re-register) a RoboTwin VLA-RL task.

    Args:
        task_id:    e.g. ``"RoboTwin-PlaceCup-GRPO-v0"``.
        env_cfg:    instance of an EnvCfg with a ``build()`` method.
        runner_cfg: instance of a VLA runner cfg (e.g. ``VLAGRPORunnerCfg``).
    """
    # Idempotent: re-running the package import (e.g. in tests) won't raise.
    gym.envs.registry.pop(task_id, None)
    gym.register(
        id=task_id,
        entry_point=_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "RoboRenForce_entry_point": runner_cfg,
        },
    )
