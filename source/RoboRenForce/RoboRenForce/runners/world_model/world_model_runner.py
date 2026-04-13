from __future__ import annotations

from importlib.resources import path
import os
import time
import torch
import pickle
from typing import Optional, Dict, List
from dataclasses import MISSING

from RoboRenForce import configclass
from RoboRenForce.components.world_models import WorldModelBase, WorldModelBaseCfg
from RoboRenForce.utils.logging import timeit
from RoboRenForce.runners.base_runner import BaseRunner, BaseRunnerCfg
from RoboRenForce.buffer import ReplayBufferBaseCfg, ReplayBufferBase


class WorldModelBasedRunner(BaseRunner):
    """Base runner for model-based algorithms.
    
    Provides common functionality for:
    - World model management
    - World model training via trainer (optional)
    - Replay buffer for world model data (optional)
    
    Subclasses should implement:
    - `init_components()`: Initialize world model and related components
    - `_learn()`: Main training loop
    """
    
    cfg: "WorldModelBasedRunnerCfg"
    flow_model: Optional["WorldModelBase"]

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.actor_critic.state_dict(),
            "obs_norm_state_dict": self.obs_normalizer.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "world_model_state_dict": self.flow_model.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        self.logger.save_model(saved_dict, path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        
        if "obs_norm_state_dict" in loaded_dict:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            print("[INFO]: Observation normalization parameters loaded successfully.")
        else:
            print("[WARNING]: Normalization parameters not found in checkpoint!")

        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        if device is not None: self.actor_critic.to(device)
        return lambda x: self.actor_critic.act_inference(self.obs_normalizer(x))

    def train_mode(self):
        self.actor_critic.train()
        self.obs_normalizer.train()
        self.critic_normalizer.train()

    def eval_mode(self):
        self.actor_critic.eval()
        self.obs_normalizer.eval()
        self.critic_normalizer.eval()


    @torch.no_grad()
    def save_real_dynamics_example(
        self,
        num_steps: int,
        out_dir: str | None = None,
    ) -> str:
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        dynamic_list: List[torch.Tensor] = []
        action_list: List[torch.Tensor] = []
        for _ in range(num_steps):
            actions = self.alg.act(obs, critic_obs)
            obs, reward, done, infos = self.env.step(actions.to(self.env.device))
            critic_obs = infos["observations"].get("critic", obs)
            obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
            (
                system_dynamic,
                system_action,
                _system_extension,
                _system_contact,
                _system_termination,
            ) = self.env.get_system_observation()
            dynamic_list.append(system_dynamic.clone())
            action_list.append(system_action.clone())
        real_dynamic = torch.stack(dynamic_list, dim=1)  # [N, T, dyn_dim]
        real_action = torch.stack(action_list, dim=1)    # [N, T, act_dim]
        num_envs, T, dyn_dim = real_dynamic.shape
        _, _, act_dim = real_action.shape
        data = {
            "real_dynamic": real_dynamic.detach().cpu().numpy(),
            "real_action": real_action.detach().cpu().numpy(),
            "dynamic_dim": dyn_dim,
            "action_dim": act_dim,
            "num_envs": num_envs,
            "num_steps": T,
            "dim_params": dict(self.env.dim_params),
        }
        if out_dir is None:
            base_dir = self.logger.log_dir or "."
            out_dir = os.path.join(base_dir, "world_model_real_examples")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"real_example_it_{self.current_learning_iteration}.pkl"
        )
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        return out_path


@configclass
class WorldModelBasedRunnerCfg(BaseRunnerCfg):
    """Base configuration for world model-based runners.
    
    Subclasses should add:
    - world_model_cfg: Configuration for the world model
    - world_model_trainer_cfg: Optional configuration for world model trainer
    - replay_buffer_cfg: Optional configuration for replay buffer
    """
    class_type: type[WorldModelBasedRunner] = WorldModelBasedRunner
    world_model_cfg: Optional[WorldModelBaseCfg] = None
