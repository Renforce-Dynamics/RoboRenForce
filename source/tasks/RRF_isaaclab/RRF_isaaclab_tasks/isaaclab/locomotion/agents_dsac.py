from RoboRenForce import configclass
from RoboRenForce import runners, algorithms, components, networks
from RoboRenForce.buffer import replay_bundle, DirectTransitionBufferCfg
import math

@configclass
class LocoRLCfgBase(runners.OffPolicyRunnerCfg):
    seed=42
    num_steps_per_env=20
    max_iterations=720_000
    save_interval=4800
    experiment_name="None"
    run_name="dsac"
    # ---- Policy: Actor-Critic for DSAC ----
    policy=components.ActorCriticPackCfg(
        actor_cfg=components.SACActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[
                    [('ReLU', {})],
                    [('ReLU', {})],
                    [('ReLU', {})],
                    []
                ]
            ),
            hidden_dim=256,
            use_tanh=True,
            log_std_min=-5.0,
            log_std_max=2.0,
            action_scale=3.14,
            action_bias=0.0
        ),
        critic_cfg=components.GaussianQNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[
                    [('LayerNorm', {}), ('ReLU', {})],
                    [('LayerNorm', {}), ('ReLU', {})],
                    [('LayerNorm', {}), ('ReLU', {})],
                    []
                ]
            )
        )
    )
    # ---- DSAC Algorithm Parameters ----
    algorithm=algorithms.DSACCfg(
        gamma=0.99, # 0.99, 0.97
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-5,
        auto_entropy=True,
        bound=True,
        alpha=math.e, # 0.2, 0.001
        target_entropy=0,  # None -> -action_dim # 0.0
        max_grad_norm=1.0,
        actor_update_freq=2,
        target_update_freq=1
    )
    # ---- Replay Buffer Configuration ----
    replay_cfg=replay_bundle.ReplayBundle(
        replay_buffer_cfg=DirectTransitionBufferCfg(
            max_steps=1_000_000,  # Large buffer for off-policy learning
            warmup_steps=100_000,
        ),
        replay_num_epoches=1,
        replay_mini_batch_size=256,
        replay_num_batch_per_epoch=100
    )
    # ---- Logger Configuration ----
    logger_cfg = runners.TqdmStyleLoggerCfg(
        logger="tensorboard",
        is_log_ep_info=False,
        is_log_update=False,
        is_log_sample=False,
        show_performance_curve=True,
        performance_history_size=25,
        width=75,
        pad=20
    )
    # ---- Normalizers ----
    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
    critic_normalize_cfg = components.NormalizerEmpiricalCfg()
