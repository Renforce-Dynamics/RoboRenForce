"""
Qwen2-VL Policy Adapter for RoboRenForce.

Auto-registers with the model registry on import.
"""

from RRF_models.registry import register_model
from .qwen2vl_policy import Qwen2VLPolicy, Qwen2VLPolicyCfg


def build_qwen2vl_policy(cfg=None, **kwargs) -> Qwen2VLPolicy:
    """Factory function for Qwen2-VL policy."""
    if cfg is None:
        raise ValueError("Qwen2VLPolicyCfg is required. Use get_model('qwen2vl', cfg=...)")
    if isinstance(cfg, dict):
        cfg = Qwen2VLPolicyCfg(**cfg)
    return Qwen2VLPolicy(cfg)


register_model("qwen2vl", build_qwen2vl_policy)
