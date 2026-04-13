import gymnasium as gym
from . import (
    agents_ppo,
    agents_smooth,
    agents_sac,
    agents_dsac,
    agents_sapg,
    agents_sacp,
    tasks,
)

names = [
    "UnitreeA1RoughEnvCfg",
    "UnitreeA1FlatEnvCfg",
    "UnitreeGo1RoughEnvCfg",
    "UnitreeGo1FlatEnvCfg",
    "UnitreeGo2RoughEnvCfg",
    "UnitreeGo2FlatEnvCfg",
    "AnymalBRoughEnvCfg",
    "AnymalBFlatEnvCfg",
    "AnymalCRoughEnvCfg",
    "AnymalCFlatEnvCfg",
    "AnymalDRoughEnvCfg",
    "AnymalDFlatEnvCfg",
    "H1RoughEnvCfg",
    "H1FlatEnvCfg",
    "G1RoughEnvCfg",
    "G1FlatEnvCfg",
]

for name in names:
    gym.register(
        id=f"RoboRenForce-{name[:-6]}-PPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RoboRenForce_entry_point": agents_ppo.LocoRLCfgBase().replace(experiment_name=f"{name[:-6]}"),
        },
    )
    
for name in names:
    gym.register(
        id=f"RoboRenForce-{name[:-6]}-SAC",
        entry_point="RoboRenForce.utils.isaaclab.envs:ManagerBasedOffRlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RoboRenForce_entry_point": agents_sac.LocoRLCfgBase().replace(experiment_name=f"{name[:-6]}"),
        },
    )

for name in names:
    gym.register(
        id=f"RoboRenForce-{name[:-6]}-DSAC",
        entry_point="RoboRenForce.utils.isaaclab.envs:ManagerBasedOffRlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RoboRenForce_entry_point": agents_dsac.LocoRLCfgBase().replace(experiment_name=f"{name[:-6]}"),
        },
    )

for name in names:
    gym.register(
        id=f"RoboRenForce-{name[:-6]}-SAPG",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RoboRenForce_entry_point": agents_sapg.LocoRLSAPGCfgBase().replace(experiment_name=f"{name[:-6]}"),
        },
    )

for name in names:
    gym.register(
        id=f"RoboRenForce-{name[:-6]}-SACP",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RoboRenForce_entry_point": agents_sacp.LocoRLCfgBase().replace(experiment_name=f"{name[:-6]}"),
        },
    )

for name in names:
    gym.register(
        id=f"RoboRenForce-{name[:-6]}-CAPSPPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RoboRenForce_entry_point": agents_smooth.LocoRLCAPSPPOCfgBase().replace(
                experiment_name=f"{name[:-6]}-CAPS"
            ),
        },
    )

for name in names:
    gym.register(
        id=f"RoboRenForce-{name[:-6]}-L2C2PPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RoboRenForce_entry_point": agents_smooth.LocoRLL2C2PPOCfgBase().replace(
                experiment_name=f"{name[:-6]}-L2C2"
            ),
        },
    )


for name in names:
    gym.register(
        id=f"RoboRenForce-{name[:-6]}-LipsPPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(tasks, name),
            "RoboRenForce_entry_point": agents_smooth.LocoRLLipsPPOCfgBase().replace(
                experiment_name=f"{name[:-6]}-LipsPPO"
            ),
        },
    )