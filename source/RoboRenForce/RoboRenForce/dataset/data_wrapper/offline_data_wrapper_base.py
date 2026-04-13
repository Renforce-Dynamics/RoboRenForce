from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from RoboRenForce import configclass
from RoboRenForce.utils.template import ClassTemplateBase, ClassTemplateBaseCfg
from RoboRenForce.dataset.data_loader import OfflineDataLoaderBase, OfflineDataLoaderBaseCfg
from dataclasses import MISSING


class OfflineDataWrapperBase(ClassTemplateBase, ABC):
    cfg: "OfflineDataWrapperBaseCfg"
    data_loader: OfflineDataLoaderBase
    
    def __init__(
        self,
        cfg: "OfflineDataWrapperBaseCfg",
        device: str = "cpu"
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.data_loader = cfg.data_loader_cfg.construct_from_cfg(device=device)
        self._trajectories = self._load_all_trajectories()
        self._dim_params = self._infer_dim_params()
    
    @property
    def dim_params(self) -> Dict[str, int]:
        return self._dim_params
    
    def _infer_dim_params(self) -> Dict[str, int]:
        if len(self._trajectories) == 0:
            raise ValueError("No trajectories found in dataset")
        
        first_traj = self._trajectories[0]
        dim_params = {}
        
        if "policy" in first_traj:
            dim_params["policy_dim"] = first_traj["policy"].shape[-1]
        else:
            raise ValueError("'policy' field is required in trajectory data")
        
        if "action" in first_traj:
            dim_params["action_dim"] = first_traj["action"].shape[-1]
        else:
            raise ValueError("'action' field is required in trajectory data")
        
        dim_params["critic_dim"] = first_traj.get("critic", first_traj["policy"]).shape[-1]
        dim_params["dynamic_dim"] = first_traj.get("dynamic", first_traj["policy"]).shape[-1]
        
        if "rewards" in first_traj:
            dim_params["rewards_dim"] = first_traj["rewards"].shape[-1]
        elif "reward" in first_traj:
            dim_params["rewards_dim"] = 1
        else:
            dim_params["rewards_dim"] = 1
        
        return dim_params
    
    @abstractmethod
    def get_batch(
        self,
        batch_size: int,
        sampling_strategy: str = "random",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        ...
    
    @abstractmethod
    def get_trajectory(self, traj_idx: int) -> Dict[str, torch.Tensor]:
        ...
    
    def __len__(self) -> int:
        return len(self._trajectories)
    
    def _load_all_trajectories(self) -> List[Dict[str, np.ndarray]]:
        # TODO for large scale datset, may be delayed loading.
        trajectories = []
        for traj in self.data_loader.load_trajectories():
            trajectories.append(traj)
        return trajectories


@configclass
class OfflineDataWrapperBaseCfg(ClassTemplateBaseCfg):
    class_type: type[OfflineDataWrapperBase] = OfflineDataWrapperBase
    data_loader_cfg: OfflineDataLoaderBaseCfg = MISSING