from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.commands_cfg import NullCommandCfg
from isaaclab.envs.mdp.commands.null_command import NullCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EpisodeSuccessCommand(NullCommand):
    """Command generator that tracks episode success rate.
    
    Inherits from NullCommand but adds episode success rate tracking functionality.
    """

    def __init__(self, cfg: NullCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the success tracking command generator."""
        super().__init__(cfg, env)
        
        # buffer to record success information
        self.cur_success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_success_buf = torch.zeros_like(self.cur_success_buf)
        
        # metrics to track success rates
        self.metrics["step_success_rate"] = torch.zeros(1, device=self.device)
        self.metrics["episode_success_rate"] = torch.zeros(1, device=self.device)
        self.metrics["alive_time_rate"] = torch.zeros(self.num_envs, device=self.device)

    def compute(self, dt: float):
        self._update_metrics()
        
        
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)
            
        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            if metric_value.ndim == 0 or (metric_value.ndim > 0 and metric_value.shape[0] == 1):
                extras[metric_name] = metric_value.item()
            else:
                extras[metric_name] = torch.mean(metric_value[env_ids]).item()
        return extras
    
    def _update_metrics(self):
        
        # get the terminations info
        reset_terminated = self._env.termination_manager.terminated # (num_envs, )
        reset_time_outs = self._env.termination_manager.time_outs # (num_envs, )
        reset_dones = self._env.termination_manager.dones # (num_envs, )

        self.metrics["step_success_rate"] = torch.sum(reset_time_outs.float()) \
            / (torch.sum(reset_dones.float()) + 1e-6)
            
        # record the success information
        self.last_success_buf[reset_dones] = self.cur_success_buf[reset_dones]
        self.cur_success_buf[reset_time_outs] = True
        self.cur_success_buf[reset_terminated] = False
        
        self.metrics["episode_success_rate"] = torch.sum(self.last_success_buf.float()) / self._env.num_envs
        
        # get the episode info 
        episode_length_buf = self._env.episode_length_buf # (num_envs, )
        max_episode_length = self._env.max_episode_length
        self.metrics["alive_time_rate"] = episode_length_buf.float() / max_episode_length
        
        
class ActionFluctuationRatioCommand(NullCommand):
    """Command generator that tracks action fluctuation ratio.
    
    Inherits from NullCommand but adds action fluctuation ratio tracking functionality.
    
    AFR = 1/T ∑‖aₜ - aₜ₋₁‖ = 1/T * sum(||a_t - a_{t-1}||)
    """

    def __init__(self, cfg: NullCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the action fluctuation ratio command generator."""
        super().__init__(cfg, env)
        
        # buffer to record action fluctuations, (num_envs, max_episode_length, action_dim)
        # self.max_episode_length = self._env.max_episode_length 
        # self.action_dim = self._env.action_manager.action_term_dim
        # self.action_buf = torch.zeros(self.num_envs, self.max_episode_length, self.action_dim, device=self.device)
        self.step_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # metrics to track action fluctuation ratio
        self.metrics["action_fluctuation_ratio"] = torch.zeros(self.num_envs, device=self.device)

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
            action_diff_abs = torch.abs(action_diff)  # (num_envs, action_dim)
            action_diff_norm = torch.norm(action_diff_abs, dim=-1)  # (num_envs,)
            
            # update cumulative AFR for valid environments
            # AFR = 1/T * sum(||a_t - a_{t-1}||)
            # We compute running average: new_avg = (old_avg * (t-1) + new_value) / t
            current_step = self.step_counter.float()
            self.metrics["action_fluctuation_ratio"][valid_envs] = (
                self.metrics["action_fluctuation_ratio"][valid_envs] * (current_step[valid_envs] - 1) + 
                action_diff_norm[valid_envs]
            ) / current_step[valid_envs]
        