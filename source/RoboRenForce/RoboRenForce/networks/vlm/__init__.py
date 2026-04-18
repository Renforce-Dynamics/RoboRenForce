"""Vision-Language Model (VLM) backbones."""

from .vlm_backbone_base import VLMBackbone, VLMBackboneCfg
from .fusion_layers import FusionLayer, FusionLayerCfg

# Qwen2VL requires transformers — import lazily
def __getattr__(name):
    if name in ("Qwen2VL", "Qwen2VLCfg", "Qwen3VL", "Qwen3VLCfg"):
        from .qwen2vl import Qwen2VL, Qwen2VLCfg
        globals()["Qwen2VL"] = Qwen2VL
        globals()["Qwen2VLCfg"] = Qwen2VLCfg
        globals()["Qwen3VL"] = Qwen2VL
        globals()["Qwen3VLCfg"] = Qwen2VLCfg
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "VLMBackbone",
    "VLMBackboneCfg",
    "Qwen2VL",
    "Qwen2VLCfg",
    "Qwen3VL",
    "Qwen3VLCfg",
    "FusionLayer",
    "FusionLayerCfg",
]
