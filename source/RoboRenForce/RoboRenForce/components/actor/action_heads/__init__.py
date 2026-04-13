"""Action heads for VLA."""

from .diffusion_action_head import DiffusionActionHead, DiffusionActionHeadCfg
from .regression_action_head import RegressionActionHead, RegressionActionHeadCfg

__all__ = [
    "DiffusionActionHead",
    "DiffusionActionHeadCfg",
    "RegressionActionHead",
    "RegressionActionHeadCfg",
]
