from RoboRenForce import configclass
from RoboRenForce import runners, algorithms, components, networks

@configclass
class LocoRLCfgBase(runners.OnPolicyRunnerCfg):
    """
    SACP (Soft Actor-Critic Compatible PPO) configuration for locomotion tasks.
    
    This configuration uses:
    - StateIndStdActor: Standard PPO actor (for training)
    - SACActor: SAC-compatible actor (learns to mimic PPO actor via distillation)
    - VNetwork: Value function network (PPO-style)
    
    Key design:
    - Primary training: Pure PPO optimization with StateIndStdActor
    - SAC compatibility: SACActor learns to mimic PPO actor through distillation
    - This ensures stable PPO training while maintaining SAC-compatible policy structure
    """
    seed = 42
    num_steps_per_env = 32  # Same as PPO for on-policy data collection
    max_iterations = 1500
    save_interval = 100
    experiment_name = "None"
    run_name = "sacp"
    
    # ---- Policy: Actor-Critic for SACP ----
    # Primary actor: StateIndStdActor (standard PPO actor, used for training)
    # SAC actor: Configured separately in algorithm.sac_actor_cfg
    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [ [] ]
            ),
            use_log_std=False  # Use std parameterization
        ),
        critic_cfg=components.VNetworkCfg(
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
    
    # ---- SACP Algorithm Parameters ----
    # Pure PPO optimization with SAC actor distillation
    algorithm = algorithms.SACPCfg(
        # Optimization
        learning_rate=1.0e-3,  # Same as standard PPO
        min_learning_rate=1e-5,
        max_learning_rate=1e-2,
        
        # PPO-specific
        clip_param=0.2,  # Standard PPO clipping parameter
        num_learning_epochs=5,  # Number of epochs per update
        num_mini_batches=4,  # Number of mini-batches per epoch
        value_loss_coef=1.0,  # Value loss coefficient
        entropy_coef=0.01,  # Standard PPO entropy regularization
        max_grad_norm=1.0,  # Gradient clipping
        
        # Standard PPO GAE
        gamma=0.99,  # Discount factor
        lam=0.95,  # GAE lambda
        use_clipped_value_loss=True,  # Clipped value loss for stability
        
        # KL control (optional, for adaptive learning rate)
        schedule="adaptive",  # "fixed" or "adaptive"
        desired_kl=0.01,  # Desired KL divergence for adaptive schedule
        
        # SAC actor configuration (for compatibility)
        sac_actor_cfg=components.SACActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[
                    [('ELU', {})],
                    [('ELU', {})],
                    [('ELU', {})],
                    []
                ]
            ),
            hidden_dim=128,  # Hidden dimension for mean/log_std heads (matches last hidden layer)
            use_tanh=True,  # Enable tanh squashing (required for SAC compatibility)
            log_std_min=-5.0,  # SAC-style log_std bounds
            log_std_max=2.0,
            action_scale=3.14,  # Match SAC action scale for locomotion
            action_bias=0.0
        ),
        
        # SAC actor distillation learning
        sac_actor_lr=3e-4,  # Learning rate for SAC actor (distillation)
        use_mse_distillation=True,  # Use MSE loss instead of KL divergence (more stable)
    )
    
    # ---- Logger Configuration ----
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
    
    # ---- Normalizers ----
    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
    critic_normalize_cfg = components.NormalizerEmpiricalCfg()
