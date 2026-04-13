"""HDF5 format data loader implementation."""

from __future__ import annotations

import h5py
import numpy as np
from typing import Dict, Generator, Any, List
from RoboRenForce import configclass
from RoboRenForce.dataset.data_loader.offline_data_loader_base import (
    OfflineDataLoaderBase,
    OfflineDataLoaderBaseCfg,
)
from dataclasses import MISSING


class HDF5DataLoader(OfflineDataLoaderBase):
    """HDF5 format data loader.
    
    Loads trajectories from HDF5 files. Each group (e.g., demo_0, demo_1, ...)
    is treated as a single trajectory.
    """
    
    cfg: "HDF5DataLoaderCfg"
    
    def __init__(self, cfg: "HDF5DataLoaderCfg", device):
        super().__init__(cfg, device)
        self._data_info = None
    
    def load_trajectories(self) -> Generator[Dict[str, np.ndarray], None, None]:
        """Load trajectories from HDF5 file.
        
        Yields:
            Dict[str, np.ndarray]: Each trajectory as a dictionary of numpy arrays.
        """
        with h5py.File(self.cfg.data_path, "r") as f:
            data_io = f[self.cfg.group_name]
            for group_name in data_io.keys():
                group = data_io[group_name]
                
                traj = {}
                for dataset_name in group.keys():
                    ds = group[dataset_name]
                    traj[dataset_name] = ds[()]  # Read as numpy array
                yield traj
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get dataset metadata information.
        
        Returns:
            Dict containing:
                - "num_trajectories": int - Number of trajectories
                - "total_steps": int - Total number of steps
                - "data_format": str - Data format description
                - "field_names": List[str] - Available field names
        """
        if self._data_info is None:
            num_trajectories = 0
            total_steps = 0
            field_names = set()
            
            with h5py.File(self.cfg.data_path, "r") as f:
                data_io = f[self.cfg.group_name]
                for group_name in data_io.keys():
                    group = data_io[group_name]
                    num_trajectories += 1
                    
                    # Get trajectory length from first dataset
                    if len(group.keys()) > 0:
                        first_key = list(group.keys())[0]
                        traj_length = group[first_key].shape[0]
                        total_steps += traj_length
                        
                        # Collect field names
                        field_names.update(group.keys())
            
            self._data_info = {
                "num_trajectories": num_trajectories,
                "total_steps": total_steps,
                "data_format": "HDF5",
                "field_names": sorted(list(field_names)),
            }
        
        return self._data_info


@configclass
class HDF5DataLoaderCfg(OfflineDataLoaderBaseCfg):
    """Configuration for HDF5 data loader."""
    
    class_type: type[HDF5DataLoader] = HDF5DataLoader
    
    data_path: str = MISSING
    """Path to the HDF5 file."""
    
    group_name: str = "data"
    """Group name in HDF5 file (default: "data")."""
