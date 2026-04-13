"""
Unified agent configurations for all 7 AFR-enabled locomotion baselines.

This module contains all agent configurations for the comprehensive action smoothness comparison:
1. Pure PPO - Standard PPO baseline
2. CAPSPPO - Consistency-based Action Policy Smoothness  
3. L2C2PPO - Lipschitz-Constrained Policy and Critic
4. LipsPPO - LipsNet++ with Lipschitz constraint regularization
5. LipsCAPSPPO - Hybrid combining CAPS mechanism with LipsPPO
6. L2C2LipsPPO - Hybrid combining L2C2 mechanism with LipsPPO
7. ActionSmooth - Reward-based action smoothness (uses Pure PPO algorithm)
"""

from RoboRenForce import configclass
from RoboRenForce import runners, algorithms, components, networks
from RoboRenForce.components.actor.lipschitz_actor import LipschitzActorCfg


# =============================================================================
# 1. Pure PPO Baseline (Standard PPO with AFR tracking)
# =============================================================================

@configclass
class LocoRLPurePPOCfg(runners.OnPolicyRunnerCfg):
    """Pure PPO configuration - Standard PPO baseline with AFR tracking."""
    
    seed = 42
    num_steps_per_env = 24  # 论文设置：24步/环境
    max_iterations = 6000   # 论文设置：充分训练
    save_interval = 500     # 调整保存间隔
    experiment_name = "None"
    run_name = "pure_ppo"
    
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


# =============================================================================
# 2. CAPSPPO - Consistency-based Action Policy Smoothness
# =============================================================================

@configclass
class LocoRLCAPSPPOCfg(runners.OnPolicyRunnerCfg):
    """CAPS-PPO configuration - Temporal and spatial smoothness regularization."""
    
    seed = 42
    num_steps_per_env = 24  # 论文设置：24步/环境
    max_iterations = 6000   # 论文设置：充分训练
    save_interval = 500     # 调整保存间隔
    experiment_name = "None"
    run_name = "caps_ppo"
    
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
        caps_lambda_t=1e-2,  # Temporal smoothness weight
        caps_lambda_s=1e-2,  # Spatial smoothness weight
        caps_sigma=5e-2,     # Smoothness kernel width
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


# =============================================================================
# 3. L2C2PPO - Lipschitz-Constrained Policy and Critic
# =============================================================================

@configclass
class LocoRLL2C2PPOCfg(runners.OnPolicyRunnerCfg):
    """L2C2-PPO configuration - Interpolated observation smoothness."""
    
    seed = 42
    num_steps_per_env = 24  # 论文设置：24步/环境
    max_iterations = 6000   # 论文设置：充分训练
    save_interval = 500     # 调整保存间隔
    experiment_name = "None"
    run_name = "l2c2_ppo"
    
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
        l2c2_lambda_pi=5e-3,   # Policy smoothness weight
        l2c2_lambda_v=2.5e-3,  # Value smoothness weight
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


# =============================================================================
# 4. LipsPPO - LipsNet++ with Lipschitz constraint regularization
# =============================================================================

@configclass
class LocoRLLipsPPOCfg(runners.OnPolicyRunnerCfg):
    """LipsNet++ PPO configuration - Lipschitz gradient norm regularization."""
    
    seed = 42
    num_steps_per_env = 24  # 论文设置：24步/环境
    max_iterations = 6000   # 论文设置：充分训练
    save_interval = 500     # 调整保存间隔
    experiment_name = "None"
    run_name = "lips_ppo"
    
    policy = components.ActorCriticPackCfg(
        actor_cfg=LipschitzActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
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
                activations=[[('ELU', {})]] * 3 + [[]]
            )
        )
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
        pad=20
    )
    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
    critic_normalize_cfg = components.NormalizerEmpiricalCfg()


# =============================================================================
# 5. LipsCAPSPPO - Hybrid combining CAPS mechanism with LipsPPO
# =============================================================================

@configclass
class LocoRLLipsCAPSPPOCfg(runners.OnPolicyRunnerCfg):
    """Hybrid LipsCAPSPPO configuration combining CAPS mechanism with LipsPPO.
    
    This hybrid approach combines:
    - CAPS: Time-based smoothness regularization (caps_lambda_t, caps_lambda_s, caps_sigma)
    - LipsPPO: Lipschitz regularization (lips_lambda) 
    - Lips-Actor: FFT filtering + Lipschitz constraints in network architecture
    """
    
    seed = 42
    num_steps_per_env = 24  # 论文设置：24步/环境
    max_iterations = 6000   # 论文设置：充分训练
    save_interval = 500     # 调整保存间隔
    experiment_name = "None"
    run_name = "lipscaps_ppo"
    
    policy = components.ActorCriticPackCfg(
        actor_cfg=LipschitzActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            ),
            init_noise_std=1.0,
            enable_fft=True,  # Lips-Actor FFT filtering
            fft_2d=False,
            obs_seq_len=1,
            fft_filter_1d_cfg=networks.FFTFilter1DCfg(
                kernel_scale=0.02
            ),
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            )
        )
    )
    
    algorithm = algorithms.LipsCAPSPPOCfg(
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
        
        # CAPS mechanism parameters
        caps_lambda_t=1e-2,  # CAPS temporal smoothness weight
        caps_lambda_s=1e-2,  # CAPS spatial smoothness weight  
        caps_sigma=5e-2,     # CAPS smoothness kernel width
        
        # LipsPPO mechanism parameters
        lips_lambda=5e-3,    # LipsNet++ Lipschitz regularization weight
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


# =============================================================================
# 6. L2C2LipsPPO - Hybrid combining L2C2 mechanism with LipsPPO
# =============================================================================

@configclass
class LocoRLL2C2LipsPPOCfg(runners.OnPolicyRunnerCfg):
    """Hybrid L2C2LipsPPO configuration combining L2C2 mechanism with LipsPPO.
    
    This hybrid approach combines:
    - L2C2: Lipschitz-constrained policy and critic regularization (l2c2_lambda_pi, l2c2_lambda_v)
    - LipsPPO: Lipschitz regularization (lips_lambda)
    - Lips-Actor: FFT filtering + Lipschitz constraints in network architecture
    """
    
    seed = 42
    num_steps_per_env = 24  # 论文设置：24步/环境
    max_iterations = 6000   # 论文设置：充分训练
    save_interval = 500     # 调整保存间隔
    experiment_name = "None"
    run_name = "l2c2lips_ppo"
    
    policy = components.ActorCriticPackCfg(
        actor_cfg=LipschitzActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            ),
            init_noise_std=1.0,
            enable_fft=True,  # Lips-Actor FFT filtering
            fft_2d=False,
            obs_seq_len=1,
            fft_filter_1d_cfg=networks.FFTFilter1DCfg(
                kernel_scale=0.02
            ),
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[('ELU', {})]] * 3 + [[]]
            )
        )
    )
    
    algorithm = algorithms.LipsL2C2PPOCfg(
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
        
        # L2C2 mechanism parameters
        l2c2_lambda_pi=5e-3,    # L2C2 λ_π: policy smoothness weight
        l2c2_lambda_v=2.5e-3,   # L2C2 λ_V: value smoothness weight
        
        # LipsPPO mechanism parameters
        lips_lambda=5e-3,       # LipsNet++ Lipschitz regularization weight
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


# =============================================================================
# 7. ActionSmooth - Reward-based action smoothness (uses Pure PPO algorithm)
# =============================================================================

# Note: ActionSmooth uses the same Pure PPO algorithm configuration
# The smoothness is implemented at the environment reward level
# So we can reuse LocoRLPurePPOCfg for ActionSmooth tasks

# Alias for clarity in task registration
LocoRLActionSmoothCfg = LocoRLPurePPOCfg


# =============================================================================
# Configuration Summary
# =============================================================================

# All 7 baseline configurations for AFR comparison:
BASELINE_CONFIGS = {
    "PurePPO": LocoRLPurePPOCfg,
    "CAPSPPO": LocoRLCAPSPPOCfg, 
    "L2C2PPO": LocoRLL2C2PPOCfg,
    "LipsPPO": LocoRLLipsPPOCfg,
    "LipsCAPSPPO": LocoRLLipsCAPSPPOCfg,
    "L2C2LipsPPO": LocoRLL2C2LipsPPOCfg,
    "ActionSmooth": LocoRLActionSmoothCfg,
}
