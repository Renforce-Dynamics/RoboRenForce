"""Dynamics locomotion tasks with custom modifications.

This module contains custom environment modifications including:
- AFR (Action Fluctuation Ratio) tracking for flat locomotion tasks
- Custom observation space modifications
"""

import gymnasium as gym
from . import afr_flat_tasks, agents_unified

# Define all AFR-enabled flat task configurations
afr_flat_task_configs = {
    "UnitreeA1Flat": afr_flat_tasks.AFRUnitreeA1FlatEnvCfg,
    "UnitreeGo1Flat": afr_flat_tasks.AFRUnitreeGo1FlatEnvCfg,
    "UnitreeGo2Flat": afr_flat_tasks.AFRUnitreeGo2FlatEnvCfg,
    "AnymalBFlat": afr_flat_tasks.AFRAnymalBFlatEnvCfg,
    "AnymalCFlat": afr_flat_tasks.AFRAnymalCFlatEnvCfg,
    "AnymalDFlat": afr_flat_tasks.AFRAnymalDFlatEnvCfg,
    "H1Flat": afr_flat_tasks.AFRH1FlatEnvCfg,
    "G1Flat": afr_flat_tasks.AFRG1FlatEnvCfg,
}

# Register AFR-enabled environments for all algorithms
# Register AFR-enabled PPO tasks
for name, task_cfg in afr_flat_task_configs.items():
    gym.register(
        id=f"RoboRenForce-AFR-{name}-PPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": task_cfg,
            "RoboRenForce_entry_point": agents_unified.LocoRLPurePPOCfg().replace(
                experiment_name=f"AFR-{name}-PPO"
            ),
        },
    )

# Register AFR-enabled CAPSPPO tasks  
for name, task_cfg in afr_flat_task_configs.items():
    gym.register(
        id=f"RoboRenForce-AFR-{name}-CAPSPPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": task_cfg,
            "RoboRenForce_entry_point": agents_unified.LocoRLCAPSPPOCfg().replace(
                experiment_name=f"AFR-{name}-CAPSPPO"
            ),
        },
    )

# Register AFR-enabled L2C2PPO tasks
for name, task_cfg in afr_flat_task_configs.items():
    gym.register(
        id=f"RoboRenForce-AFR-{name}-L2C2PPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": task_cfg,
            "RoboRenForce_entry_point": agents_unified.LocoRLL2C2PPOCfg().replace(
                experiment_name=f"AFR-{name}-L2C2PPO"
            ),
        },
    )

# Register AFR-enabled LipsPPO tasks
for name, task_cfg in afr_flat_task_configs.items():
    gym.register(
        id=f"RoboRenForce-AFR-{name}-LipsPPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": task_cfg,
            "RoboRenForce_entry_point": agents_unified.LocoRLLipsPPOCfg().replace(
                experiment_name=f"AFR-{name}-LipsPPO"
            ),
        },
    )

# Register AFR-enabled LipsCAPSPPO tasks (Hybrid: CAPS + LipsPPO + Lips-Actor)
for name, task_cfg in afr_flat_task_configs.items():
    gym.register(
        id=f"RoboRenForce-AFR-{name}-LipsCAPSPPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": task_cfg,
            "RoboRenForce_entry_point": agents_unified.LocoRLLipsCAPSPPOCfg().replace(
                experiment_name=f"AFR-{name}-LipsCAPSPPO"
            ),
        },
    )

# Register AFR-enabled L2C2LipsPPO tasks (Hybrid: L2C2 + LipsPPO + Lips-Actor)
for name, task_cfg in afr_flat_task_configs.items():
    gym.register(
        id=f"RoboRenForce-AFR-{name}-L2C2LipsPPO",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": task_cfg,
            "RoboRenForce_entry_point": agents_unified.LocoRLL2C2LipsPPOCfg().replace(
                experiment_name=f"AFR-{name}-L2C2LipsPPO"
            ),
        },
    )

# Import action smooth reward tasks
from . import action_smooth_reward_tasks

# Register AFR + ActionSmooth tasks (Reward-based action smoothness baseline with AFR tracking)
for name, task_cfg in action_smooth_reward_tasks.action_smooth_afr_flat_task_configs.items():
    gym.register(
        id=f"RoboRenForce-AFR-{name}-ActionSmooth",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": task_cfg,
            "RoboRenForce_entry_point": agents_unified.LocoRLActionSmoothCfg().replace(
                experiment_name=f"AFR-{name}-ActionSmooth"
            ),
        },
    )

# Pure PPO + AFR tasks are already registered above (lines 24-36)

print(f"[INFO] Registered {len(afr_flat_task_configs) * 7} AFR-enabled locomotion tasks in dynamics/locomotion.")
