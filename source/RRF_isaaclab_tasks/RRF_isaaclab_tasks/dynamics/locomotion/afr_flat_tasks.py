from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.commands_cfg import NullCommandCfg
from isaaclab.envs.mdp.commands.null_command import NullCommand

# Import all flat environment configurations
from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.flat_env_cfg import UnitreeGo1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_b.flat_env_cfg import AnymalBFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ActionFluctuationRatioCommand(NullCommand):
    """Command generator that tracks action fluctuation ratio.
    
    Inherits from NullCommand but adds action fluctuation ratio tracking functionality.
    
    AFR = 1/T ∑‖aₜ - aₜ₋₁‖ = 1/T * sum(||a_t - a_{t-1}||)
    """

    def __init__(self, cfg: NullCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the action fluctuation ratio command generator."""
        super().__init__(cfg, env)
        
        # buffer to record action fluctuations
        self.step_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # metrics to track action fluctuation ratio
        self.metrics["action_fluctuation_ratio"] = torch.zeros(self.num_envs, device=self.device)
        
        # Create a dummy command tensor to satisfy environment wrapper requirements
        self._command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return dummy command tensor for compatibility with environment wrapper."""
        return self._command

    def compute(self, dt: float):
        self._update_metrics()
        self.step_counter += 1
        
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)
            
        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0
        
        # reset the buffer
        self.step_counter[env_ids] = 0

        return extras
    
    def _update_metrics(self):
        # get valid environments (not the first step)
        valid_envs = self.step_counter > 0
        
        if torch.any(valid_envs):
            # get the current and previous actions info
            cur_action = self._env.action_manager._action # (num_envs, action_dim)
            prev_action = self._env.action_manager._prev_action # (num_envs, action_dim)
            
            # compute action differences and their norms
            action_diff = cur_action - prev_action  # (num_envs, action_dim)
            action_diff_norm = torch.norm(action_diff, dim=-1)  # (num_envs,)
            
            # update cumulative AFR for valid environments
            # AFR = 1/T * sum(||a_t - a_{t-1}||)
            # We compute running average: new_avg = (old_avg * (t-1) + new_value) / t
            current_step = self.step_counter.float()
            self.metrics["action_fluctuation_ratio"][valid_envs] = (
                self.metrics["action_fluctuation_ratio"][valid_envs] * (current_step[valid_envs] - 1) + 
                action_diff_norm[valid_envs]
            ) / current_step[valid_envs]


@configclass
class ActionFluctuationRatioCommandCfg(NullCommandCfg):
    """Configuration for action fluctuation ratio command."""
    class_type: type = ActionFluctuationRatioCommand


def add_afr_tracking(target_cfg_class):
    """Add AFR tracking to an Isaac Lab environment configuration."""
    original_name = target_cfg_class.__name__
    original_module = target_cfg_class.__module__
    
    class AFRTrackedCfg(target_cfg_class):
        def __post_init__(self):
            if hasattr(super(), '__post_init__'):
                super().__post_init__()
            
            # Add AFR command tracking
            self.commands.afr_tracker = ActionFluctuationRatioCommandCfg()
    
    AFRTrackedCfg.__name__ = f"AFR{original_name}"
    AFRTrackedCfg.__qualname__ = f"AFR{original_name}"
    AFRTrackedCfg.__module__ = original_module
    return AFRTrackedCfg


# Create AFR-enabled versions of all flat environment configurations
AFRUnitreeA1FlatEnvCfg = add_afr_tracking(UnitreeA1FlatEnvCfg)
AFRUnitreeGo1FlatEnvCfg = add_afr_tracking(UnitreeGo1FlatEnvCfg)
AFRUnitreeGo2FlatEnvCfg = add_afr_tracking(UnitreeGo2FlatEnvCfg)
AFRAnymalBFlatEnvCfg = add_afr_tracking(AnymalBFlatEnvCfg)
AFRAnymalCFlatEnvCfg = add_afr_tracking(AnymalCFlatEnvCfg)
AFRAnymalDFlatEnvCfg = add_afr_tracking(AnymalDFlatEnvCfg)
AFRH1FlatEnvCfg = add_afr_tracking(H1FlatEnvCfg)
AFRG1FlatEnvCfg = add_afr_tracking(G1FlatEnvCfg)

# Export all AFR-enabled configurations
__all__ = [
    "ActionFluctuationRatioCommand",
    "ActionFluctuationRatioCommandCfg", 
    "add_afr_tracking",
    "AFRUnitreeA1FlatEnvCfg",
    "AFRUnitreeGo1FlatEnvCfg", 
    "AFRUnitreeGo2FlatEnvCfg",
    "AFRAnymalBFlatEnvCfg",
    "AFRAnymalCFlatEnvCfg",
    "AFRAnymalDFlatEnvCfg",
    "AFRH1FlatEnvCfg",
    "AFRG1FlatEnvCfg",
]
