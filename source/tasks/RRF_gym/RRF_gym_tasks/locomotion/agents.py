"""
Agent configurations for classic Gymnasium continuous-control tasks.

Provides ready-to-use runner configs for PPO, SAC, DSAC, DSACT
on MuJoCo locomotion and other continuous-action envs.
"""

import math
from RoboRenForce import configclass
from RoboRenForce.runners import OnPolicyRunnerCfg, OffPolicyRunnerCfg, LoggerBaseCfg
from RoboRenForce.components.normalizer import NormalizerBaseCfg
from RoboRenForce.components.actor_critic_pack import ActorCriticPackCfg
from RoboRenForce.components.actor import SACActorCfg, StateIndStdActorCfg
from RoboRenForce.components.critic import MultiQNetworkCfg, VNetworkCfg, GaussianQNetworkCfg
from RoboRenForce.components import ModuleList
from RoboRenForce.algorithms.off_policy.sac import SACCfg
from RoboRenForce.algorithms.off_policy.dsac import DSACCfg, DSACTCfg
from RoboRenForce.algorithms.on_policy.ppo import PPOCfg
from RoboRenForce.buffer import replay_bundle, DirectTransitionBufferCfg
from RoboRenForce.networks.mlp import MLPCfg


# ============================================================================
# PPO
# ============================================================================

@configclass
class GymPPOCfg(OnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 128
    max_iterations = 200_000
    save_interval = 200
    experiment_name = ""
    run_name = "ppo"
    policy = ActorCriticPackCfg(
        actor_cfg=StateIndStdActorCfg(
            backbone_cfg=MLPCfg(
                hidden_features=[128, 128],
                activations=[[("ELU", {})]] * 3,
            ),
            use_log_std=True,
        ),
        critic_cfg=VNetworkCfg(
            backbone_cfg=MLPCfg(
                hidden_features=[128, 128],
                activations=[[("ELU", {})]] * 3,
            ),
        ),
    )
    algorithm = PPOCfg(
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        value_loss_coef=0.25,
        entropy_coef=0.01,
        num_learning_epochs=10,
        num_mini_batches=8,
        learning_rate=3e-4,
        schedule="fixed",
        max_grad_norm=1.0,
        desired_kl=0.02,
    )
    logger_cfg = LoggerBaseCfg(logger="tensorboard")
    obs_normalize_cfg = NormalizerBaseCfg()
    critic_normalize_cfg = NormalizerBaseCfg()


# ============================================================================
# SAC
# ============================================================================

@configclass
class GymSACCfg(OffPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 20
    max_iterations = 200_000
    save_interval = 10_000
    experiment_name = ""
    run_name = "sac"
    policy = ActorCriticPackCfg(
        actor_cfg=SACActorCfg(
            backbone_cfg=MLPCfg(
                hidden_features=[256, 256, 256],
                activations=[[("ReLU", {})]] * 4,
            ),
            use_tanh=True,
            log_std_min=-5,
            log_std_max=2,
            action_bias=0.0,
            action_scale=1.0,
        ),
        critic_cfg=MultiQNetworkCfg(
            num_q=2,
            backbone_cfg=MLPCfg(
                hidden_features=[256, 256, 256],
                activations=[[("ReLU", {})]] * 3 + [[]],
            ),
        ),
    )
    algorithm = SACCfg(
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-5,
        auto_entropy=True,
        alpha=math.e,
        max_grad_norm=5.0,
        actor_update_freq=2,
        target_update_freq=1,
    )
    replay_cfg = replay_bundle.ReplayBundle(
        replay_buffer_cfg=DirectTransitionBufferCfg(
            max_steps=1_000_000,
            warmup_steps=10_000,
        ),
        replay_num_epoches=1,
        replay_mini_batch_size=256,
        replay_num_batch_per_epoch=100,
    )
    logger_cfg = LoggerBaseCfg(logger="tensorboard", is_log_sample=False)
    obs_normalize_cfg = NormalizerBaseCfg()
    critic_normalize_cfg = NormalizerBaseCfg()


# ============================================================================
# DSAC
# ============================================================================

@configclass
class GymDSACCfg(OffPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 20
    max_iterations = 200_000
    save_interval = 10_000
    experiment_name = ""
    run_name = "dsac"
    policy = ActorCriticPackCfg(
        actor_cfg=SACActorCfg(
            backbone_cfg=MLPCfg(
                hidden_features=[256, 256, 256],
                activations=[[("GELU", {})]] * 4,
            ),
            use_tanh=True,
            log_std_min=-20,
            log_std_max=1,
            action_bias=0.0,
            action_scale=1.0,
        ),
        critic_cfg=GaussianQNetworkCfg(
            backbone_cfg=MLPCfg(
                hidden_features=[256, 256, 256],
                activations=[[("GELU", {})]] * 3 + [[]],
            ),
        ),
    )
    algorithm = DSACCfg(
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-5,
        auto_entropy=True,
        bound=True,
        alpha=math.e,
        max_grad_norm=5.0,
        actor_update_freq=2,
        target_update_freq=2,
    )
    replay_cfg = replay_bundle.ReplayBundle(
        replay_buffer_cfg=DirectTransitionBufferCfg(
            max_steps=1_000_000,
            warmup_steps=1_000,
        ),
        replay_num_epoches=1,
        replay_mini_batch_size=256,
        replay_num_batch_per_epoch=100,
    )
    logger_cfg = LoggerBaseCfg(logger="tensorboard", is_log_sample=False)
    obs_normalize_cfg = NormalizerBaseCfg()
    critic_normalize_cfg = NormalizerBaseCfg()


# ============================================================================
# DSACT (twin distributional)
# ============================================================================

@configclass
class GymDSACTCfg(OffPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 20
    max_iterations = 200_000
    save_interval = 10_000
    experiment_name = ""
    run_name = "dsact"
    policy = ActorCriticPackCfg(
        actor_cfg=SACActorCfg(
            backbone_cfg=MLPCfg(
                hidden_features=[256, 256, 256],
                activations=[[("GELU", {})]] * 4,
            ),
            use_tanh=True,
            log_std_min=-20,
            log_std_max=1,
            action_bias=0.0,
            action_scale=1.0,
        ),
        critic_cfg=ModuleList(
            module_list=[
                GaussianQNetworkCfg(
                    backbone_cfg=MLPCfg(
                        hidden_features=[256, 256, 256],
                        activations=[[("GELU", {})]] * 3 + [[]],
                    ),
                ),
                GaussianQNetworkCfg(
                    backbone_cfg=MLPCfg(
                        hidden_features=[256, 256, 256],
                        activations=[[("GELU", {})]] * 3 + [[]],
                    ),
                ),
            ],
        ),
    )
    algorithm = DSACTCfg(
        gamma=0.99,
        tau=0.005,
        tau_b=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-5,
        auto_entropy=True,
        alpha=math.e,
        max_grad_norm=5.0,
        actor_update_freq=2,
        target_update_freq=2,
    )
    replay_cfg = replay_bundle.ReplayBundle(
        replay_buffer_cfg=DirectTransitionBufferCfg(
            max_steps=1_000_000,
            warmup_steps=10_000,
        ),
        replay_num_epoches=1,
        replay_mini_batch_size=256,
        replay_num_batch_per_epoch=100,
    )
    logger_cfg = LoggerBaseCfg(logger="tensorboard", is_log_sample=False)
    obs_normalize_cfg = NormalizerBaseCfg()
    critic_normalize_cfg = NormalizerBaseCfg()
