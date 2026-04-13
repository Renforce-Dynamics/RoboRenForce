"""离线数据加载器基类

提供从不同格式文件读取数据的统一接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Generator, Any, Optional, List
import numpy as np
from RoboRenForce import configclass
from RoboRenForce.utils.template import ClassTemplateBase, ClassTemplateBaseCfg
from dataclasses import MISSING


class OfflineDataLoaderBase(ClassTemplateBase, ABC):    
    cfg: "OfflineDataLoaderBaseCfg"
    
    def __init__(self, cfg, device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
    
    @abstractmethod
    def load_trajectories(self) -> Generator[Dict[str, np.ndarray], None, None]:
        ...
    
    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        ...
    
    def get_first_trajectory(self) -> Optional[Dict[str, np.ndarray]]:
        try:
            return next(self.load_trajectories())
        except StopIteration:
            return None


@configclass
class OfflineDataLoaderBaseCfg(ClassTemplateBaseCfg):
    class_type: type[OfflineDataLoaderBase] = OfflineDataLoaderBase
    data_path: str = MISSING
