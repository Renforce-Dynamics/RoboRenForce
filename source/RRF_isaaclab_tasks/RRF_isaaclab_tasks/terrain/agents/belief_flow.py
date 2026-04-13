from RoboRenForce import configclass
from RoboRenForce import runners, algorithms, components, networks
from RoboRenForce.runners.world_model.flow_model_runner import FlowModelRunnerCfg


@configclass
class A1BeliefFlowCfg(FlowModelRunnerCfg):
    """Belief + flow-model runner config for Unitree A1 rough terrain task.

    All runner, policy, algorithm and logging hyper-parameters are defined here.
    World-model-related configs (flow model, belief updater, belief_dim, trainer)
    inherit sane defaults from `FlowModelRunnerCfg`.
    """

    # ---- generic runner settings ----
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 500
    experiment_name = "UnitreeA1Rough"
    run_name = "flow_model"

    # ---- policy: actor with built-in belief encoder + critic ----
    policy = components.ActorCriticPackCfg(
        actor_cfg=components.BeliefEncoderActorCfg(
            encoder_cfg=components.VecStateEncoderCfg(
                backbone_cfg=networks.MLPCfg(
                    hidden_features=[256, 128],
                    activations=[[("ELU", {})]] * 3,
                )
            ),
            actor_cfg=components.StateIndStdActorCfg(
                backbone_cfg=networks.MLPCfg(
                    hidden_features=[512, 256, 128],
                    activations=[[("ELU", {})]] * 3 + [[]],
                ),
                use_log_std=False,
            ),
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[("ELU", {})]] * 3 + [[]],
            )
        ),
    )

    # ---- PPO algorithm hyper-parameters ----
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

    world_model_cfg: components.world_models.BeliefFlowModelCfg = components.world_models.BeliefFlowModelCfg(
        backbone_cfg=networks.MLPCfg(
            hidden_features=[256, 256],
            activations=[[('ReLU', {})], [('ReLU', {})], [],]
        )
    )
    
    belief_updater_cfg: components.world_models.BeliefUpdaterCfg = components.world_models.BeliefUpdaterCfg(
        backbone_cfg=networks.MLPCfg(
            hidden_features=[256, 256],
            activations=[[('ReLU', {})], [('ReLU', {})], [],]
        )
    )
    belief_dim: int = 64
    flow_model_trainer_cfg: algorithms.world_model_trainer.FlowModelTrainerCfg = \
        algorithms.world_model_trainer.FlowModelTrainerCfg(
            flow_learning_rate = 1e-3,
            flow_weight_decay = 0.0,
            obs_update_learning_rate = 1e-3,
            obs_update_weight_decay = 0.0,
            num_mini_batches = 1,
            mini_batch_size = 512,
            max_grad_norm = 1.0,
        )


    # ---- logging & normalization ----
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
