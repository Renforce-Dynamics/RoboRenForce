"""SAC agent configurations for MJLab locomotion tasks."""

from RoboRenForce import configclass
from RoboRenForce import runners, algorithms, components, networks


@configclass
class MJLabLocoSACCfg(runners.OffPolicyRunnerCfg):
    """SAC configuration for MJLab locomotion tasks."""

    seed = 42
    max_iterations = 10000
    save_interval = 500
    experiment_name = "None"
    run_name = "mjlab_sac"

    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            ),
            use_log_std=True
        ),
        critic_cfg=components.QNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            )
        )
    )

    algorithm = algorithms.SACCfg(
        actor_lr=3.0e-4,
        critic_lr=3.0e-4,
        alpha_lr=3.0e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy=True,
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
        pad=20
    )
    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
    critic_normalize_cfg = components.NormalizerEmpiricalCfg()
