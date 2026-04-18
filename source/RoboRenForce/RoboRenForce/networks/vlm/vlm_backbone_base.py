"""
VLM Backbone Base Class

Base class for all Vision-Language Model backbones (System 2).

TODO Phase 1.2 (Week 1, Priority P0):
- [ ] Define base VLM interface
- [ ] Add freeze/unfreeze methods
- [ ] Define forward signature
- [ ] Support LoRA integration
"""

from typing import Optional
from dataclasses import MISSING

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


@configclass
class VLMBackboneCfg(ModuleBaseCfg):
    """Base configuration for VLM backbones."""

    class_type: type["VLMBackbone"] = MISSING

    # Model
    model_name: str = MISSING

    # Training
    freeze: bool = True

    # Output
    output_dim: int = MISSING  # VL feature dimension


class VLMBackbone(ModuleBase):
    """
    Base class for VLM backbones.

    Responsibilities:
    - Load pretrained VLM from HuggingFace or local
    - Extract vision-language features (System 2)
    - Support freezing for efficient training
    - Optional LoRA fine-tuning

    Reference: .references/Psi0/src/psi/models/vlm_wrapper.py
    """

    def __init__(self, cfg: VLMBackboneCfg):
        super().__init__()
        self.cfg = cfg

        self.model_name = cfg.model_name
        self.freeze = cfg.freeze
        self.output_dim = cfg.output_dim

        # TODO: Subclasses should:
        # - Load pretrained model
        # - Apply freezing if specified
        # - Setup LoRA (if needed)

    def forward(
        self,
        image: torch.Tensor,
        text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract VL features.

        Args:
            image: (B, C, H, W) image tensor
            text: (B, max_text_len) token IDs or None

        Returns:
            vl_features: (B, output_dim) VL feature tensor

        TODO:
        - Subclasses must implement this
        - Extract features from VLM
        - Pool or select relevant features
        - Return fixed-size feature vector
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def freeze_backbone(self):
        """
        Freeze all VLM parameters.

        TODO:
        - Set requires_grad=False for all parameters
        - Set model to eval mode
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze_backbone(self):
        """
        Unfreeze VLM parameters.

        TODO:
        - Set requires_grad=True for all parameters
        - Set model to train mode
        """
        for param in self.parameters():
            param.requires_grad = True
        self.train()
