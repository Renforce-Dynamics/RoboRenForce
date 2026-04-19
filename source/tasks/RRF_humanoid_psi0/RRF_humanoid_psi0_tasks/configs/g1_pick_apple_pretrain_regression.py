"""
G1 PickApple — Pretrain with Regression Head

Pretraining config for Psi0 G1 PickApple demonstrations.
Uses regression action head (simplest baseline).

Usage (single GPU):
    python scripts/vla/pretrain/train_single_gpu.py \
        --config RRF_humanoid_psi0_tasks.configs.g1_pick_apple_pretrain_regression

Usage (DDP):
    torchrun --nproc_per_node=4 scripts/vla/pretrain/train_ddp.py \
        --config RRF_humanoid_psi0_tasks.configs.g1_pick_apple_pretrain_regression
"""

from RoboRenForce.runners.vla.pretrain.vla_pretrain_runner import VLAPretrainRunnerCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import RegressionActionHeadCfg
from RoboRenForce.algorithms.vla_training.pretrain_algorithm import VLAPretrainAlgorithmCfg
from RoboRenForce.prototype.embodied import MixtureDatasetCfg

from RRF_humanoid_psi0_tasks.datasets import Psi0G1PickAppleCfg
from RRF_humanoid_psi0_tasks.robots import UnitreeG1Cfg


def get_config(
    data_root: str = "",
    frames_dir: str = "",
    use_mock_vlm: bool = True,
):
    """Build pretrain config for G1 PickApple with regression head."""
    robot_cfg = UnitreeG1Cfg()

    dataset_entry = Psi0G1PickAppleCfg()
    if data_root:
        dataset_entry.data_root = data_root
    if frames_dir:
        dataset_entry.frames_dir = frames_dir

    mixture_cfg = MixtureDatasetCfg(
        entries=[dataset_entry],
        mix_mode="concat",
        unified_action_dim=robot_cfg.action_dim,
    )

    action_head_cfg = RegressionActionHeadCfg(
        hidden_dims=[256, 256],
        activation="relu",
        action_dim=robot_cfg.action_dim,
        action_horizon=1,
    )

    algorithm_cfg = VLAPretrainAlgorithmCfg(
        action_loss_weight=1.0,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
    )

    runner_cfg = VLAPretrainRunnerCfg(
        dataset_cfg=mixture_cfg,
        algorithm_cfg=algorithm_cfg,
        batch_size=32,
        num_epochs=5,
        num_workers=4,
        save_interval=500,
        log_interval=10,
        checkpoint_dir="checkpoints/g1_pick_apple_pretrain",
    )

    return {
        "runner_cfg": runner_cfg,
        "robot_cfg": robot_cfg,
        "action_head_cfg": action_head_cfg,
        "algorithm_cfg": algorithm_cfg,
        "dataset_entry": dataset_entry,
        "mixture_cfg": mixture_cfg,
    }
