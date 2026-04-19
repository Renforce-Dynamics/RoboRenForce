"""
G1 Dex3 PickApple VLA Pretraining Configuration

Single-dataset pretraining on Psi0 G1 PickApple demonstrations.

Usage (single GPU, mock VLM):
    python scripts/vla/pretrain/train_single_gpu.py \
        --config RRF_vla_tasks.vla_pretrain.g1_pick_apple

Usage (DDP):
    torchrun --nproc_per_node=4 scripts/vla/pretrain/train_ddp.py \
        --config RRF_vla_tasks.vla_pretrain.g1_pick_apple

This config demonstrates how to compose:
    Robot (RRF_vla_tasks.robots) + Dataset (RRF_vla_tasks.datasets)
    + Model (RoboRenForce.networks) + Algorithm (RoboRenForce.algorithms)
"""

from RoboRenForce.runners.vla.pretrain.vla_pretrain_runner import VLAPretrainRunnerCfg
from RoboRenForce.components.actor.vla_actor import VLAActorCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import RegressionActionHeadCfg
from RoboRenForce.algorithms.vla_training.pretrain_algorithm import VLAPretrainAlgorithmCfg
from RoboRenForce.prototype.embodied_ai import MixtureDatasetCfg

from RRF_vla_tasks.datasets import Psi0G1PickAppleCfg
from RRF_vla_tasks.robots import UnitreeG1Cfg


def get_config(
    data_root: str = "",
    frames_dir: str = "",
    use_mock_vlm: bool = True,
):
    """
    Build the experiment config.

    Args:
        data_root: Override data path (empty = use default from entry)
        frames_dir: Pre-extracted frames directory
        use_mock_vlm: If True, use MockVLM for fast testing
    """
    robot_cfg = UnitreeG1Cfg()

    # --- Dataset entry ---
    dataset_entry = Psi0G1PickAppleCfg()
    if data_root:
        dataset_entry.data_root = data_root
    if frames_dir:
        dataset_entry.frames_dir = frames_dir

    # MixtureDataset wrapping a single entry (extensible to multi-dataset)
    mixture_cfg = MixtureDatasetCfg(
        entries=[dataset_entry],
        mix_mode="concat",
        unified_action_dim=robot_cfg.action_dim,  # 36
    )

    # --- VLM backbone ---
    if use_mock_vlm:
        from RoboRenForce.networks.vlm.vlm_backbone_base import VLMBackboneCfg
        # MockVLM will be handled by the training script
        vlm_cfg = None
    else:
        # Real VLM — uncomment when ready
        # from RoboRenForce.networks.vlm.qwen2vl import Qwen2VLCfg
        # vlm_cfg = Qwen2VLCfg(
        #     model_name="Qwen/Qwen2-VL-2B-Instruct",
        #     freeze=True,
        #     output_dim=1536,
        # )
        raise NotImplementedError("Real VLM not yet integrated")

    # --- Action head ---
    action_head_cfg = RegressionActionHeadCfg(
        hidden_dims=[256, 256],
        activation="relu",
        action_dim=robot_cfg.action_dim,       # 36
        action_horizon=1,
    )

    # --- Algorithm ---
    algorithm_cfg = VLAPretrainAlgorithmCfg(
        action_loss_weight=1.0,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
    )

    # --- Runner ---
    runner_cfg = VLAPretrainRunnerCfg(
        dataset_cfg=mixture_cfg,
        algorithm_cfg=algorithm_cfg,
        batch_size=32,
        num_epochs=5,
        num_workers=4,
        save_interval=500,
        log_interval=10,
        checkpoint_dir="checkpoints/g1_pick_apple",
    )
    # Note: vla_actor_cfg is set by the training script after determining
    # VLM output dim (mock vs real). This keeps the config VLM-agnostic.
    # runner_cfg.vla_actor_cfg = VLAActorCfg(...)

    return {
        "runner_cfg": runner_cfg,
        "robot_cfg": robot_cfg,
        "action_head_cfg": action_head_cfg,
        "algorithm_cfg": algorithm_cfg,
        "dataset_entry": dataset_entry,
        "mixture_cfg": mixture_cfg,
    }
