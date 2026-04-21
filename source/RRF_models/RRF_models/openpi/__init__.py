"""
OpenPI (pi0/pi0.5) Policy Adapter

Factory + registration for Physical Intelligence's pi0 VLA model.

Usage:
    from RRF_models.registry import get_model

    policy = get_model("openpi", cfg=OpenPIPolicyCfg(
        actor_cfg=VLAActorCfg(
            vlm_backbone_cfg=OpenPICfg(model_name="lerobot/pi05_base"),
            ...
        ),
    ))
"""

from RRF_models.registry import register_model
from .openpi_policy import OpenPIPolicy, OpenPIPolicyCfg


def build_openpi_policy(cfg=None, **kwargs) -> OpenPIPolicy:
    """Factory function for OpenPI policy."""
    if cfg is None:
        raise ValueError(
            "OpenPIPolicyCfg is required. Usage:\n"
            "  get_model('openpi', cfg=OpenPIPolicyCfg(actor_cfg=...))"
        )
    if isinstance(cfg, dict):
        cfg = OpenPIPolicyCfg(**cfg)
    return OpenPIPolicy(cfg)


register_model("openpi", build_openpi_policy)
