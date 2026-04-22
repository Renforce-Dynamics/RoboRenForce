"""Utility to load MJLab task configurations from the mjlab task registry."""

import importlib


def load_mjlab_env_cfg(task_name: str, play: bool = False):
    """Load environment config from the MJLab task registry.

    Args:
        task_name: MJLab task ID (e.g., "Mjlab-Velocity-Flat-Unitree-Go1").
        play: If True, return play (evaluation) config.

    Returns:
        ManagerBasedRlEnvCfg instance.
    """
    from mjlab.tasks.registry import load_env_cfg
    return load_env_cfg(task_name, play=play)


def load_mjlab_rl_cfg(task_name: str):
    """Load RL runner config from the MJLab task registry.

    Args:
        task_name: MJLab task ID.

    Returns:
        RslRlBaseRunnerCfg instance.
    """
    from mjlab.tasks.registry import load_rl_cfg
    return load_rl_cfg(task_name)


def make_mjlab_env(task_name: str, num_envs: int = 4096, device: str = "cuda:0",
                   play: bool = False, wrapper: str = "dynamic"):
    """Create a MJLab env wrapped for RoboRenForce.

    Args:
        task_name: MJLab task ID.
        num_envs: Number of parallel environments.
        device: Compute device.
        play: Use play config.
        wrapper: One of "base", "dynamic", "group".

    Returns:
        Wrapped environment.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from .vecenv_wrapper import RoboRenForceMJLabEnvWrapper
    from .dynamic_env_wrapper import MJLabDynamicEnvWrapper
    from .group_vec_wrapper import MJLabGroupVecWrapper

    cfg = load_mjlab_env_cfg(task_name, play=play)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=device)

    wrapper_map = {
        "base": RoboRenForceMJLabEnvWrapper,
        "dynamic": MJLabDynamicEnvWrapper,
        "group": MJLabGroupVecWrapper,
    }
    wrapper_cls = wrapper_map.get(wrapper, MJLabDynamicEnvWrapper)
    return wrapper_cls(env)
