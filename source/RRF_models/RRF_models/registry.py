"""
Model Registry — Factory + registration for pretrained policy adapters.

Each policy adapter registers itself via register_model(). Models are
lazily imported to avoid loading heavy dependencies until needed.

Reference: RLinf rlinf/models/__init__.py
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from RoboRenForce.prototype.embodied import BasePolicy


# Type alias for model builder functions
ModelBuilder = Callable[..., BasePolicy]

# Global registry: model_type -> builder function
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(model_type: str, builder: ModelBuilder):
    """Register a model builder function.

    Each policy package calls this in its __init__.py:
        register_model("qwen2vl", build_qwen2vl_policy)
    """
    if model_type in _MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_type}' is already registered")
    _MODEL_REGISTRY[model_type] = builder


def get_model(model_type: str, cfg: Any = None, **kwargs) -> BasePolicy:
    """Instantiate a policy by model_type string.

    Args:
        model_type: registered model name (e.g. "qwen2vl", "mlp_baseline")
        cfg: model-specific config (configclass or dict)
        **kwargs: additional args passed to builder

    Returns:
        BasePolicy instance
    """
    # Lazy registration: try to import the model package if not yet registered
    if model_type not in _MODEL_REGISTRY:
        _try_lazy_import(model_type)

    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Unknown model type '{model_type}'. "
            f"Available: {available}. "
            f"Did you install the required extras? (pip install RRF_models[{model_type}])"
        )

    builder = _MODEL_REGISTRY[model_type]
    return builder(cfg=cfg, **kwargs)


def list_models() -> list[str]:
    """Return list of registered model type names."""
    # Trigger lazy imports for all known model packages
    for name in _KNOWN_MODELS:
        if name not in _MODEL_REGISTRY:
            _try_lazy_import(name)
    return sorted(_MODEL_REGISTRY.keys())


# ---- Lazy import helpers ----

# Known model packages — add new ones here
_KNOWN_MODELS = {
    "qwen2vl": "RRF_models.qwen2vl",
    "qwen3vl": "RRF_models.qwen3vl",
    "openpi": "RRF_models.openpi",
    "gr00t": "RRF_models.gr00t",
    "mlp_baseline": "RRF_models.mlp_baseline",
}


def _try_lazy_import(model_type: str):
    """Try to import a model package to trigger its register_model() call."""
    module_path = _KNOWN_MODELS.get(model_type)
    if module_path is None:
        return
    try:
        __import__(module_path)
    except ImportError:
        pass  # Dependency not installed, silently skip
