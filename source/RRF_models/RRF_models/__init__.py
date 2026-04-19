"""
RRF Models — Pretrained model adapters for RoboRenForce.

Wraps open-source pretrained models (Qwen2-VL, OpenPI, GR00T, etc.)
behind the BasePolicy interface defined in RoboRenForce core.

This package has heavy dependencies (transformers, peft, etc.) and is
intentionally separate from the core algorithm library.

Usage:
    from RRF_models import get_model, list_models

    policy = get_model("qwen2vl", cfg)
    actions = policy.predict_action(obs)
"""

from RRF_models.registry import get_model, register_model, list_models

__all__ = ["get_model", "register_model", "list_models"]
