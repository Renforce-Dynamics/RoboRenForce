from __future__ import annotations

import os
from abc import abstractmethod
from typing import Dict
import torch
from RoboRenForce import configclass
from dataclasses import MISSING

from RoboRenForce.runners.base_runner import BaseRunner, BaseRunnerCfg
from RoboRenForce.dataset.data_wrapper import OfflineDataWrapperBase, OfflineDataWrapperBaseCfg
from RoboRenForce.dataset.data_loader import OfflineDataLoaderBase, OfflineDataLoaderBaseCfg
from RoboRenForce.runners.logger import LoggerBaseCfg


class OfflineRunnerBase(BaseRunner):    
    cfg: "OfflineRunnerBaseCfg"
    data_cli: "OfflineDataWrapperBase"
    def __init__(
        self,
        train_cfg: "OfflineRunnerBaseCfg",
        log_dir=None,
        device: str = "cpu",
    ):
        super().__init__(
            train_cfg=train_cfg,
            env=None,
            log_dir=log_dir,
            device=device
        )

    def init_components(self):
        self.data_cli = self.cfg.data_cli_cfg.construct_from_cfg(device=self.device)
    
    def learn(self):
        self.logger.init_logger()
        self.train_mode()
        
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + self.cfg.max_iterations
        
        for it in range(start_iter, tot_iter):
            sample_infos = self.gather()
            alg_update_infos = self.update()
            
            if self.logger.log_dir is not None:
                log_locals = locals()
                self.logger.log(self, log_locals)
            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            self.current_learning_iteration = it
        
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
    
    @abstractmethod
    def gather(self) -> Dict[str, float]:
        ...
    
    @abstractmethod
    def update(self) -> Dict[str, float]:
        ...
    
    def save(self, path, infos=None):
        saved_dict = {
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.logger.log_dir is not None:
            self.logger.save_model(saved_dict, path, self.current_learning_iteration)
    
    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos", None)
    
    def train_mode(self):
        self.train()
    
    def eval_mode(self):
        self.eval()


@configclass
class OfflineRunnerBaseCfg(BaseRunnerCfg):
    class_type          : type[OfflineRunnerBase] = OfflineRunnerBase
    data_cli_cfg        : OfflineDataWrapperBaseCfg = MISSING
    num_steps_per_env   : int = None

