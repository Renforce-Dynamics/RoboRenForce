from RoboRenForce import configclass
from RoboRenForce import runners, algorithms, components, networks, buffer
from RoboRenForce.networks.transformer import TransformerBackboneCfg


@configclass
class PPOCfg(runners.OnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 32
    max_iterations = 3125
    save_interval = 100
    experiment_name = "None"
    run_name = "ppo"
    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[256, 128, 64], activations=[[("ELU", {})]] * 3 + [[]]
            ),
            use_log_std=False,
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[256, 128, 64], activations=[[("ELU", {})]] * 3 + [[]]
            )
        ),
    )
    algorithm = algorithms.PPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    logger_cfg = runners.TqdmStyleLoggerCfg(
        logger="tensorboard",
        is_log_ep_info=False,
        is_log_update=False,
        is_log_sample=False,
        show_performance_curve=True,
        performance_history_size=75,
        width=75,
        pad=20,
    )
    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
    critic_normalize_cfg = components.NormalizerEmpiricalCfg()


@configclass
class MBPOCfg(runners.MBPOOnPolicyRunnerCfg):
    # ---- generic runner settings ----
    seed = 42
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 100
    experiment_name = "None"
    run_name = "mbpo"

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
            )
        ),
    )
    algorithm = algorithms.MBPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # ---- MBPO runner ----
    # warmup
    frozen_wramup = False
    system_dynamics_warmup_iterations = 1500
    imagination_num_envs = 0
    imagination_num_steps_per_env = 16
    system_dynamics_trainer_cfg = (
        algorithms.world_model_trainer.SystemDynamicsTrainerCfg(
            dynamic_update_forecast_horizon=8,
            dynamic_update_num_mini_batches=4,
            dynamic_update_mini_batch_size=1024,
        )
    )
    system_dynamics_cfg = components.world_models.SystemDynamicsTransformerCfg(
        history_horizon=8,
        backbone_cfg=TransformerBackboneCfg(
            dim=256,
            num_layers=2,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.1,
            max_seq_len=32,
            positional_encoding="learned",
            causal=False,
            use_final_layer_norm=True,
        ),
    )
    replay_buffer_cfg = (
        buffer.direct_based.dynamic_replay_buffer.DynamicReplayBufferCfg(
            buffer_size=32 * 128
        )
    )
    logger_cfg = runners.TqdmStyleLoggerCfg(
        logger="tensorboard",
        is_log_ep_info=False,
        is_log_update=False,
        is_log_sample=False,
        show_performance_curve=True,
        performance_history_size=75,
        width=75,
        pad=20,
    )
    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
    critic_normalize_cfg = components.NormalizerEmpiricalCfg()
