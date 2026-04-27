"""GRPO runner cfg for CALVIN D-split."""

from __future__ import annotations

from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithmCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import (
    RegressionActionHeadCfg,
)
from RoboRenForce.components.actor.vla_actor import VLAActorCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.networks.vlm.qwen2vl import Qwen2VLCfg
from RoboRenForce.runners.vla.rl.vla_grpo_runner import VLAGRPORunnerCfg
from RoboRenForce.utils.configclass import configclass


def _build_policy():
    from RRF_models.qwen2vl import build_qwen2vl_policy
    from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicyCfg

    cfg = Qwen2VLPolicyCfg(
        actor_cfg=VLAActorCfg(
            vlm_backbone_cfg=Qwen2VLCfg(
                model_name="Qwen/Qwen2-VL-2B-Instruct",
                freeze=True, device_map_auto=False,
            ),
            freeze_vlm=True,
            fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
            action_head_cfg=RegressionActionHeadCfg(
                action_dim=7, action_horizon=1, hidden_dims=[256, 256],
            ),
            use_proprioception=True, use_text=True,
        ),
        use_value_head=False,
        proprio_dim=7,
    )
    return build_qwen2vl_policy(cfg=cfg)


@configclass
class CalvinDSplitGRPOCfg(VLAGRPORunnerCfg):
    grpo_cfg = GRPOAlgorithmCfg(
        group_size=8, clip_ratio_low=0.2, clip_ratio_high=0.28,
        kl_beta=0.05, update_epochs=4, learning_rate=1e-4, reward_coef=1.0,
    )
    log_interval: int = 1
    save_interval: int = 50
    checkpoint_dir: str = "checkpoints/CALVIN-D-GRPO-v0"
    eval_interval: int = 50
    eval_episodes: int = 20

    build_policy = staticmethod(_build_policy)
