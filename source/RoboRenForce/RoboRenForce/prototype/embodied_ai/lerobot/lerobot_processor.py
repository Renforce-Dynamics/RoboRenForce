"""
LeRobot Data Processor

Preprocessing pipeline for LeRobot dataset samples.

IO Contract:
    Input  (__call__):
        raw_data: dict with keys like:
            "observation.state"  : (proprio_dim,) numpy/tensor
            "action"             : (action_dim,) numpy/tensor
            "observation.image.*": (H, W, C) numpy/PIL (optional)

    Output (__call__):
        processed dict with same keys but:
            images  -> (C, H', W') float32 tensor, normalized
            state   -> (proprio_dim,) float32 tensor, normalized
            action  -> (action_dim,) float32 tensor, normalized
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torchvision.transforms as T
import numpy as np

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


class LeRobotProcessor(ModuleBase):
    """
    Data preprocessing pipeline for LeRobot samples.

    Applies:
    - Image: resize + normalize (ImageNet stats)
    - Proprioception: (x - mean) / (std + eps)
    - Action: (x - mean) / (std + eps)
    """

    def __init__(self, cfg: LeRobotProcessorCfg, stats: Optional[dict] = None):
        super().__init__()
        self.cfg = cfg
        self.stats = stats or {}

        # Build image transform pipeline
        transforms = [
            T.ToPILImage(),
            T.Resize(cfg.image_size),
            T.ToTensor(),  # -> (C, H, W) in [0, 1]
            T.Normalize(mean=cfg.image_mean, std=cfg.image_std),
        ]
        if cfg.augment_images:
            aug = T.ColorJitter(
                brightness=cfg.augment_brightness,
                contrast=cfg.augment_contrast,
            )
            transforms.insert(2, aug)

        self.image_transform = T.Compose(transforms)

        self.image_transform_eval = T.Compose([
            T.ToPILImage(),
            T.Resize(cfg.image_size),
            T.ToTensor(),
            T.Normalize(mean=cfg.image_mean, std=cfg.image_std),
        ])

    def __call__(self, raw_data: Dict, training: bool = True) -> Dict:
        processed = {}
        transform = self.image_transform if training else self.image_transform_eval

        for key, value in raw_data.items():
            if "image" in key:
                processed[key] = self.process_image(value, transform)
            elif key == "action":
                t = self._to_tensor(value)
                if self.cfg.normalize_actions:
                    t = self.normalize(t, key)
                processed[key] = t
            elif key in ("observation.state", "proprioception"):
                t = self._to_tensor(value)
                if self.cfg.normalize_proprioception:
                    t = self.normalize(t, key)
                processed[key] = t
            elif isinstance(value, (int, float, str)):
                processed[key] = value
            else:
                processed[key] = self._to_tensor(value)

        return processed

    def process_image(self, image, transform=None) -> torch.Tensor:
        if transform is None:
            transform = self.image_transform

        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] in (1, 3):
                return T.Compose([
                    T.Resize(self.cfg.image_size),
                    T.Normalize(mean=self.cfg.image_mean, std=self.cfg.image_std),
                ])(image.float())
            elif image.ndim == 3:
                image = image.numpy().astype(np.uint8)

        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

        return transform(image)

    def normalize(self, data: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.stats:
            return data
        stat = self.stats[key]
        mean = self._to_tensor(stat["mean"])
        std = self._to_tensor(stat["std"])
        return (data - mean) / (std + 1e-8)

    def denormalize(self, data: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.stats:
            return data
        stat = self.stats[key]
        mean = self._to_tensor(stat["mean"])
        std = self._to_tensor(stat["std"])
        return data * (std + 1e-8) + mean

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, (int, float)):
            return torch.tensor(x, dtype=torch.float32)
        if isinstance(x, list):
            return torch.tensor(x, dtype=torch.float32)
        return x


@configclass
class LeRobotProcessorCfg(ModuleBaseCfg):
    """Data preprocessing configuration."""

    class_type: type[LeRobotProcessor] = LeRobotProcessor

    image_size: tuple = (224, 224)
    image_mean: list = [0.485, 0.456, 0.406]
    image_std: list = [0.229, 0.224, 0.225]

    normalize_actions: bool = True
    normalize_proprioception: bool = True

    augment_images: bool = False
    augment_brightness: float = 0.2
    augment_contrast: float = 0.2
