"""
Qwen3-VL Policy Adapter for RoboRenForce.

Auto-registers with the model registry on import.
"""

from RRF_models.registry import register_model
from .qwen3vl_policy import Qwen3VLPolicy, Qwen3VLPolicyCfg


def build_qwen3vl_policy(cfg=None, **kwargs) -> Qwen3VLPolicy:
    """Factory function for Qwen3-VL policy."""
    if cfg is None:
        raise ValueError("Qwen3VLPolicyCfg is required. Use get_model('qwen3vl', cfg=...)")
    if isinstance(cfg, dict):
        cfg = Qwen3VLPolicyCfg(**cfg)
    return Qwen3VLPolicy(cfg)


register_model("qwen3vl", build_qwen3vl_policy)
