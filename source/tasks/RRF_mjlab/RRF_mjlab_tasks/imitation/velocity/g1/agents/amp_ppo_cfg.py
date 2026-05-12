"""AMP PPO runner config for Unitree G1 velocity AMP task (MJLab).

Reference: beyondAMP/source/amp_tasks_mjlab/amp_tasks_mjlab/velocity/g1/agents/amp_ppo_cfg.py
"""

from __future__ import annotations

from RoboRenForce import configclass
from RoboRenForce import components, networks
from RoboRenForce.runners.imitation.adversarial.amp_on_policy_runner import (
    AMPOnPolicyImitationRunnerCfg,
)
from RoboRenForce.algorithms.imitation.adverserial import AMPPPOCfg

# MotionDataset lives under RRF_isaaclab_tasks today but is backend-agnostic
# (uses ``env.scene[...]`` + ``find_bodies`` which both backends expose).
from RRF_isaaclab_tasks.env_wrapper.adversarial_wrapper import MotionDatasetCfg

from ..amp_env_cfg import (
    G1_AMP_OBS_TERMS,
    G1_ANCHOR_NAME,
    G1_KEY_BODY_NAMES,
)


# Default motion-clip path. Override at run-time via the agent cfg.
G1_DEFAULT_MOTION_FILES: list[str] = []


@configclass
class MJLabG1AMPRunnerCfg(AMPOnPolicyImitationRunnerCfg):
    """AMP-PPO runner config for MJLab G1 velocity AMP."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "MJLab-G1Velocity-AMP"
    run_name = "amp"

    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[("ELU", {})]] * 3 + [[]],
            ),
            use_log_std=False,
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[("ELU", {})]] * 3 + [[]],
            ),
        ),
    )

    algorithm = AMPPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_replay_buffer_size=100_000,
        amp_reward_coef=0.5,
        amp_discr_hidden_dims=(256, 256),
        amp_task_reward_lerp=0.3,
    )

    amp_data = MotionDatasetCfg(
        asset_name="robot",
        motion_files=G1_DEFAULT_MOTION_FILES,
        body_names=G1_KEY_BODY_NAMES,
        amp_obs_terms=G1_AMP_OBS_TERMS,
        anchor_name=G1_ANCHOR_NAME,
    )

    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
