"""
MLP Baseline Policy — lightweight testing policy.

Auto-registers with the model registry on import.
"""

from RRF_models.registry import register_model
from .mlp_policy import MLPBaselinePolicy


def build_mlp_baseline(cfg=None, **kwargs) -> MLPBaselinePolicy:
    """Factory function for MLP baseline policy."""
    if cfg is None:
        cfg = {}
    if isinstance(cfg, dict):
        return MLPBaselinePolicy(**cfg)
    # Support configclass-style objects
    return MLPBaselinePolicy(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        action_horizon=getattr(cfg, "action_horizon", 1),
        hidden_dims=getattr(cfg, "hidden_dims", (256, 256)),
        use_value_head=getattr(cfg, "use_value_head", False),
    )


register_model("mlp_baseline", build_mlp_baseline)
