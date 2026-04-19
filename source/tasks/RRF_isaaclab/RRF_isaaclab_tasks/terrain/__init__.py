import gymnasium as gym
from . import task, agents

gym.register(
    id="RoboRenForce-UnitreeA1Belief-FlowModel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": task.UnitreeA1BeliefCfg,
        "RoboRenForce_entry_point": agents.A1BeliefFlowCfg().replace(
            experiment_name="UnitreeA1Belief-FlowModel"
        ),
    },
)
