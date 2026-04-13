from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg

# Import AFR tasks to extend them with action smoothness
from .afr_flat_tasks import (
    AFRUnitreeA1FlatEnvCfg,
    AFRUnitreeGo1FlatEnvCfg, 
    AFRUnitreeGo2FlatEnvCfg,
    AFRAnymalBFlatEnvCfg,
    AFRAnymalCFlatEnvCfg,
    AFRAnymalDFlatEnvCfg,
    AFRH1FlatEnvCfg,
    AFRG1FlatEnvCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_smoothness_penalty(env: ManagerBasedRLEnv, lambda_smooth: float = -0.1) -> torch.Tensor:
    """
    Compute action smoothness penalty reward term.
    
    This function implements the Action Smooth baseline method by penalizing
    large changes in actions between consecutive time steps.
    
    The action smoothness penalty is calculated as:
    r_smooth = λ_smooth * ||a_t - a_{t-1}||, where λ_smooth ≤ 0
    
    Args:
        env: The environment instance
        lambda_smooth: Action smoothness penalty weight (should be ≤ 0)
        
    Returns:
        smoothness_penalty: Penalty for action changes [num_envs]
    """
    # Get current and previous actions
    current_actions = env.action_manager._action  # [num_envs, action_dim]
    prev_actions = env.action_manager._prev_action  # [num_envs, action_dim]
    
    # Calculate L2 norm of action differences: ||a_t - a_{t-1}||
    action_diff = current_actions - prev_actions
    action_diff_norm = torch.norm(action_diff, dim=-1)  # [num_envs]
    
    # Apply smoothness penalty: r_smooth = λ_smooth * ||a_t - a_{t-1}||
    # Note: λ_smooth should be ≤ 0 to penalize large action changes
    smoothness_penalty = lambda_smooth * action_diff_norm
    
    return smoothness_penalty


# Create Action Smooth + AFR enabled environment configurations
@configclass
class ActionSmoothAFRUnitreeA1FlatEnvCfg(AFRUnitreeA1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Add action smoothness penalty to rewards
        self.rewards.action_smoothness = RewardTermCfg(
            func=action_smoothness_penalty,
            weight=1.0,  # Weight is handled inside the function
            params={"lambda_smooth": -0.1}
        )


@configclass
class ActionSmoothAFRUnitreeGo1FlatEnvCfg(AFRUnitreeGo1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Add action smoothness penalty to rewards
        self.rewards.action_smoothness = RewardTermCfg(
            func=action_smoothness_penalty,
            weight=1.0,  # Weight is handled inside the function
            params={"lambda_smooth": -0.1}
        )


@configclass
class ActionSmoothAFRUnitreeGo2FlatEnvCfg(AFRUnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Add action smoothness penalty to rewards
        self.rewards.action_smoothness = RewardTermCfg(
            func=action_smoothness_penalty,
            weight=1.0,  # Weight is handled inside the function
            params={"lambda_smooth": -0.1}
        )


@configclass
class ActionSmoothAFRAnymalBFlatEnvCfg(AFRAnymalBFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Add action smoothness penalty to rewards
        self.rewards.action_smoothness = RewardTermCfg(
            func=action_smoothness_penalty,
            weight=1.0,  # Weight is handled inside the function
            params={"lambda_smooth": -0.1}
        )


@configclass
class ActionSmoothAFRAnymalCFlatEnvCfg(AFRAnymalCFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Add action smoothness penalty to rewards
        self.rewards.action_smoothness = RewardTermCfg(
            func=action_smoothness_penalty,
            weight=1.0,  # Weight is handled inside the function
            params={"lambda_smooth": -0.1}
        )


@configclass
class ActionSmoothAFRAnymalDFlatEnvCfg(AFRAnymalDFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Add action smoothness penalty to rewards
        self.rewards.action_smoothness = RewardTermCfg(
            func=action_smoothness_penalty,
            weight=1.0,  # Weight is handled inside the function
            params={"lambda_smooth": -0.1}
        )


@configclass
class ActionSmoothAFRH1FlatEnvCfg(AFRH1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Add action smoothness penalty to rewards
        self.rewards.action_smoothness = RewardTermCfg(
            func=action_smoothness_penalty,
            weight=1.0,  # Weight is handled inside the function
            params={"lambda_smooth": -0.1}
        )


@configclass
class ActionSmoothAFRG1FlatEnvCfg(AFRG1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Add action smoothness penalty to rewards
        self.rewards.action_smoothness = RewardTermCfg(
            func=action_smoothness_penalty,
            weight=1.0,  # Weight is handled inside the function
            params={"lambda_smooth": -0.1}
        )


# Dictionary of all Action Smooth + AFR flat task configurations
action_smooth_afr_flat_task_configs = {
    "UnitreeA1Flat": ActionSmoothAFRUnitreeA1FlatEnvCfg,
    "UnitreeGo1Flat": ActionSmoothAFRUnitreeGo1FlatEnvCfg,
    "UnitreeGo2Flat": ActionSmoothAFRUnitreeGo2FlatEnvCfg,
    "AnymalBFlat": ActionSmoothAFRAnymalBFlatEnvCfg,
    "AnymalCFlat": ActionSmoothAFRAnymalCFlatEnvCfg,
    "AnymalDFlat": ActionSmoothAFRAnymalDFlatEnvCfg,
    "H1Flat": ActionSmoothAFRH1FlatEnvCfg,
    "G1Flat": ActionSmoothAFRG1FlatEnvCfg,
}
