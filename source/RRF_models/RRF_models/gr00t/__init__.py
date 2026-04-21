"""
NVIDIA GR00T N1.7 Policy Adapter

Factory + registration for NVIDIA's GR00T robot foundation model.

Usage:
    from RRF_models.registry import get_model

    policy = get_model("gr00t", cfg=GR00TPolicyCfg(
        actor_cfg=VLAActorCfg(
            vlm_backbone_cfg=GR00TCfg(model_name="nvidia/GR00T-N1.7-3B"),
            ...
        ),
    ))
"""

from RRF_models.registry import register_model
from .gr00t_policy import GR00TPolicy, GR00TPolicyCfg


def build_gr00t_policy(cfg=None, **kwargs) -> GR00TPolicy:
    """Factory function for GR00T policy."""
    if cfg is None:
        raise ValueError(
            "GR00TPolicyCfg is required. Usage:\n"
            "  get_model('gr00t', cfg=GR00TPolicyCfg(actor_cfg=...))"
        )
    if isinstance(cfg, dict):
        cfg = GR00TPolicyCfg(**cfg)
    return GR00TPolicy(cfg)


register_model("gr00t", build_gr00t_policy)
