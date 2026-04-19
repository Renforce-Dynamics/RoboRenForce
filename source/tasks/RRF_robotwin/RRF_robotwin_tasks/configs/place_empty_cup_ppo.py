"""
Config: place_empty_cup — PPO RL Fine-tuning

RL fine-tuning on RoboTwin's place_empty_cup task using PPO.
Requires a pretrained checkpoint as initialization.

Usage:
    python -m RoboRenForce.runners.vla.rl.run --config \\
        source/tasks/RRF_robotwin/RRF_robotwin_tasks/configs/place_empty_cup_ppo.py \\
        --checkpoint checkpoints/place_empty_cup_pretrain/checkpoint_final.pt
"""

from RRF_robotwin_tasks.envs.robotwin_env import RoboTwinTaskConfig


def get_config():
    env_cfg = {
        "task_config": RoboTwinTaskConfig(
            task_name="place_empty_cup",
            planner_backend="mplib",
            embodiment=["piper", "piper", 0.6],
            step_lim=200,
        ),
        "image_size": (224, 224),
        "action_dim": 14,
        "state_dim": 14,
        "max_episode_steps": 200,
        "use_custom_reward": True,
        "use_rel_reward": True,
        "reward_coef": 5.0,
        "center_crop": True,
    }

    return {
        "env_cfg": env_cfg,
        "num_envs": 64,
        "model_type": "qwen2vl",
        # TODO: complete when VLA RL runner is implemented
    }
