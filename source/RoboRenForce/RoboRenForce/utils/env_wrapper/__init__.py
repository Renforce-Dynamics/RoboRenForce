"""Submodule defining the environment definitions.

Canonical locations:
- VecEnv abstract: RoboRenForce.prototype.gym
- IsaacLab wrappers: RRF_isaaclab_tasks.env_wrapper
"""

from RoboRenForce.prototype.gym import RoboRenForceVecEnv

# Lazy import for lab_wrapper — now lives in RRF_isaaclab_tasks
def __getattr__(name):
    if name == "lab_wrapper":
        try:
            from RRF_isaaclab_tasks import env_wrapper as lab_wrapper
            return lab_wrapper
        except ImportError:
            raise ImportError(
                "lab_wrapper has moved to RRF_isaaclab_tasks.env_wrapper. "
                "Install RRF_isaaclab_tasks or update your imports."
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["RoboRenForceVecEnv"]
