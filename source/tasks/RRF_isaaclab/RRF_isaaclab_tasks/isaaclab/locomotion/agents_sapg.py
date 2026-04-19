from RoboRenForce import configclass
from RoboRenForce import runners, algorithms, components, networks
from RoboRenForce.algorithms.on_policy.epo.exploration_coefficient_epo import EPOExplorationCoefficientCfg


@configclass
class LocoRLSAPGCfgBase(runners.SAPGOnPolicyRunnerCfg):
    """
    SAPG configuration for Isaac Lab locomotion tasks.

    This configuration mirrors the PPO baseline (`agents_ppo.LocoRLCfgBase`) but:
    - Uses `SAPGPPO` as the algorithm;
    - Uses `SAPGOnPolicyRunner` as the runner;
    - Adds exploration coefficient configuration for SAPG.
    """

    # Runner basics
    seed: int = 42
    num_steps_per_env: int = 32
    max_iterations: int = 1500
    save_interval: int = 100
    experiment_name: str = "None"
    run_name: str = "sapg"

    # Policy (same network as PPO baseline)
    policy: components.ActorCriticPackCfg = components.ActorCriticPackCfg(
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

    # Algorithm: SAPG-PPO variant
    algorithm: algorithms.SAPGPPOCfg = algorithms.SAPGPPOCfg(
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

    # SAPG exploration configuration
    exploration_coef_cfg: "algorithms.on_policy.sapg.ExplorationCoefficientCfg" = (
        algorithms.on_policy.sapg.ExplorationCoefficientCfg(
            expl_type="mixed_expl_learn_param",  # scalar embedding per block
            num_blocks=4,
            embd_size=1,
            embd_init_range=(0.5, -0.5),
            reward_coef_type="entropy",
            reward_coef_scale=0.005,
            reward_coef_range=(0.5, 0.0),
        )
    )

    # SAPG augmentation behaviour
    off_policy_ratio: float = 1.0  # repeat 1 additional block
    use_leader_follower: bool = True  # use all blocks' data
    use_batch_augmentation: bool = True

    # Normalizers (same as PPO baseline)
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
class LocoRLEPOCfgBase(runners.EPOOnPolicyRunnerCfg):
    """
    EPO configuration for Isaac Lab locomotion tasks.

    This configuration extends SAPG with EPO-style block-level evolution:
    - Uses `SAPGPPO` as the algorithm (same as SAPG);
    - Uses `EPOOnPolicyRunner` as the runner;
    - Uses `EPOExplorationCoefficient` for exploration coefficient management;
    - Periodically evolves block parameters based on performance.

    EPO parameters are tuned for locomotion tasks:
    - `epo_interval_steps`: Evolution happens every 10M env steps
    - `epo_warmup_steps`: Wait 50M steps before first evolution
    - These values can be adjusted based on task scale and training duration.
    """

    # Runner basics
    seed: int = 42
    num_steps_per_env: int = 32
    max_iterations: int = 1500
    save_interval: int = 100
    experiment_name: str = "None"
    run_name: str = "epo"

    # Policy (same network as SAPG/PPO baseline)
    policy: components.ActorCriticPackCfg = components.ActorCriticPackCfg(
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

    # Algorithm: SAPG-PPO variant (same as SAPG)
    algorithm: algorithms.SAPGPPOCfg = algorithms.SAPGPPOCfg(
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

    # EPO exploration configuration
    # Note: EPOExplorationCoefficientCfg is used by default in EPOOnPolicyRunnerCfg
    # We can override it here to customize block settings
    exploration_coef_cfg: EPOExplorationCoefficientCfg = EPOExplorationCoefficientCfg(
        expl_type="mixed_expl_learn_param",  # scalar embedding per block
        num_blocks=4,  # Can be increased (e.g., 8, 16, 64) for more diversity
        embd_size=1,
        embd_init_range=(0.5, -0.5),
        reward_coef_type="entropy",
        reward_coef_scale=0.005,
        reward_coef_range=(0.5, 0.0),
    )

    # SAPG augmentation behaviour (same as SAPG)
    off_policy_ratio: float = 1.0  # repeat 1 additional block
    use_leader_follower: bool = True  # use all blocks' data
    use_batch_augmentation: bool = True

    # EPO-specific parameters
    # Tuned for locomotion tasks with ~200M total env steps (32 * 4096 * 1500)
    epo_interval_steps: int = 10_000_000
    """Interval (in env frames) between EPO evolution events.
    
    For locomotion tasks with ~200M total steps, this allows ~20 evolution events.
    Can be adjusted based on task scale:
    - Smaller tasks: 5_000_000 - 10_000_000
    - Larger tasks: 20_000_000 - 50_000_000
    """
    
    epo_warmup_steps: int = 50_000_000
    """Warmup frames before EPO starts modifying block parameters.
    
    Allows the policy to stabilize before evolution begins.
    Should be less than total training steps.
    For ~200M total steps, 50M warmup allows ~15 evolution events.
    """

    # Normalizers (same as PPO/SAPG baseline)
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

