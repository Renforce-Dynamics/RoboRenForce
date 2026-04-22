"""PPO agent configurations for MJLab locomotion tasks.

Mirrors the IsaacLab agent configs but tuned for MJLab velocity environments.
"""

from RoboRenForce import configclass
from RoboRenForce import runners, algorithms, components, networks


@configclass
class MJLabLocoPPOCfg(runners.OnPolicyRunnerCfg):
    """Standard PPO configuration for MJLab locomotion tasks.

    Hyperparameters match mjlab's default Go1/G1 velocity PPO configs:
    - 512-256-128 MLP with ELU activations
    - 24 steps per env, lr=1e-3, adaptive schedule
    """

    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "None"
    run_name = "mjlab_ppo"

    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            ),
            use_log_std=False
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            )
        )
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
        pad=20
    )
    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
    critic_normalize_cfg = components.NormalizerEmpiricalCfg()


@configclass
class MJLabLocoG1PPOCfg(MJLabLocoPPOCfg):
    """PPO config for G1 humanoid — longer training, obs normalization on."""

    max_iterations = 30000
    run_name = "mjlab_g1_ppo"
