"""GRPO runner cfg for LIBERO spatial."""

from __future__ import annotations

from RoboRenForce.runners.vla.rl.vla_grpo_runner import VLAGRPORunnerCfg
from RoboRenForce.utils.configclass import configclass

from .._shared.presets import libero_grpo_default, libero_vla_actor


def _build_policy():
    from RRF_models.qwen2vl import build_qwen2vl_policy
    from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicyCfg

    cfg = Qwen2VLPolicyCfg(
        actor_cfg=libero_vla_actor(action_dim=7),
        use_value_head=False,
        proprio_dim=7,
    )
    return build_qwen2vl_policy(cfg=cfg)


@configclass
class LiberoSpatialGRPOCfg(VLAGRPORunnerCfg):
    grpo_cfg = libero_grpo_default()
    log_interval: int = 1
    save_interval: int = 50
    checkpoint_dir: str = "checkpoints/LIBERO-Spatial-GRPO-v0"
    eval_interval: int = 50
    eval_episodes: int = 20

    build_policy = staticmethod(_build_policy)
