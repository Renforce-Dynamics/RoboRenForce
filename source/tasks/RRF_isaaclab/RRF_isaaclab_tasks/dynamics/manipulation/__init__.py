import gymnasium as gym

try:
    from . import agents, pickplace_tasks
except ImportError as e:
    import warnings
    warnings.warn(
        f"[dynamics/manipulation] Skipping task registration — missing module: {e}. "
        "FrankaCubeLift/FrankaReach tasks will not be available."
    )
    pickplace_tasks = None

if pickplace_tasks is not None:
    from . import agents

    task_names = [
        "FrankaCubeLiftEnvCfg",
        "FrankaReachEnvCfg",
    ]

    for name in task_names:
        gym.register(
            id=f"RoboRenForce-{name[:-6]}-PPO",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": getattr(pickplace_tasks, name),
                "RoboRenForce_entry_point": agents.PPOCfg().replace(
                    experiment_name=f"{name[:-6]}"
                ),
            },
        )

    for name in task_names:
        gym.register(
            id=f"RoboRenForce-{name[:-6]}-MBPO",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": getattr(pickplace_tasks, name),
                "RoboRenForce_entry_point": agents.MBPOCfg().replace(
                    experiment_name=f"{name[:-6]}"
                ),
            },
        )
