"""Vision-Language Model (VLM) backbones."""

from .vlm_backbone_base import VLMBackbone, VLMBackboneCfg
from .qwen2vl import Qwen2VL, Qwen2VLCfg
from .fusion_layers import FusionLayer, FusionLayerCfg

# Keep Qwen3VL imports for backwards compatibility (points to Qwen2VL for now)
Qwen3VL = Qwen2VL
Qwen3VLCfg = Qwen2VLCfg

__all__ = [
    "VLMBackbone",
    "VLMBackboneCfg",
    "Qwen2VL",
    "Qwen2VLCfg",
    "Qwen3VL",  # Alias for Qwen2VL
    "Qwen3VLCfg",  # Alias for Qwen2VLCfg
    "FusionLayer",
    "FusionLayerCfg",
]
