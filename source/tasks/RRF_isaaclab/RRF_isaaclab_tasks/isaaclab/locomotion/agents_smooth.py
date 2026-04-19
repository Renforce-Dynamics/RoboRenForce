from RoboRenForce import configclass
from RoboRenForce import runners, algorithms, components, networks
from RoboRenForce.components.actor.lipschitz_actor import LipschitzActorCfg


@configclass
class LocoRLCAPSPPOCfgBase(runners.OnPolicyRunnerCfg):
    """CAPS-PPO configuration for Isaac Lab locomotion tasks."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "None"
    run_name = "caps_ppo"

    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]],
            ),
            use_log_std=False,
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]],
            )
        ),
    )

    algorithm = algorithms.CAPSPPOCfg(
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
        caps_lambda_t=1e-2,
        caps_lambda_s=1e-2,
        caps_sigma=5e-2,
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
class LocoRLL2C2PPOCfgBase(runners.OnPolicyRunnerCfg):
    """L2C2-PPO configuration for Isaac Lab locomotion tasks."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "None"
    run_name = "l2c2_ppo"

    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]],
            ),
            use_log_std=False,
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]],
            )
        ),
    )

    algorithm = algorithms.L2C2PPOCfg(
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
        l2c2_lambda_pi=5e-3,
        l2c2_lambda_v=2.5e-3,
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
class LocoRLLipsPPOCfgBase(runners.OnPolicyRunnerCfg):
    """LipsNet++ PPO configuration for Isaac Lab locomotion tasks."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "None"
    run_name = "lips_ppo"

    policy = components.ActorCriticPackCfg(
        actor_cfg=LipschitzActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[
                    [('ELU', {})]
                ] * 3 + [[]],
            ),
            init_noise_std=1.0,
            enable_fft=True,
            fft_2d=False,
            obs_seq_len=1,
            fft_filter_1d_cfg=networks.FFTFilter1DCfg(
                kernel_scale=0.02
            ),
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[
                    [('ELU', {})]
                ] * 3 + [[]],
            )
        ),
    )

    algorithm = algorithms.LipsPPOCfg(
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
        lips_lambda=5e-3,  # LipsNet++ λ_Lips = 5×10^-3 from Table I
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
