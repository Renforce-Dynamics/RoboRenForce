"""
Config: place_empty_cup — PPO/GRPO RL Fine-tuning

RL fine-tuning on RoboTwin's place_empty_cup task.
Requires a pretrained checkpoint as initialization.

Usage:
    # GRPO (default):
    python scripts/vla/rl/train_robotwin_grpo.py \\
        --task place_empty_cup --num_envs 8 --iterations 100

    # PPO with value head:
    python scripts/vla/rl/train_robotwin_grpo.py \\
        --algo ppo --task place_empty_cup --num_envs 8

    # With pretrained checkpoint:
    python scripts/vla/rl/train_robotwin_grpo.py \\
        --checkpoint checkpoints/vla_pretrain/checkpoint_final.pt \\
        --task place_empty_cup

Environment requirements:
    export ASSETS_PATH=/path/to/RoboTwin
    export PYTHONPATH=/path/to/RoboTwin:$PYTHONPATH
    # SAPIEN 3 requires Vulkan GPU rendering
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

    grpo_cfg = {
        "group_size": 8,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.28,
        "learning_rate": 1e-4,
        "update_epochs": 4,
        "kl_beta": 0.05,
        "reward_coef": 1.0,
    }

    ppo_cfg = {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.28,
        "value_loss_coef": 0.5,
        "learning_rate": 1e-4,
        "update_epochs": 4,
        "kl_beta": 0.0,
    }

    return {
        "env_cfg": env_cfg,
        "num_envs": 8,
        "model_type": "qwen2vl",
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "freeze_vlm": True,
        "grpo_cfg": grpo_cfg,
        "ppo_cfg": ppo_cfg,
        "iterations": 100,
        "checkpoint_dir": "checkpoints/rl/place_empty_cup",
        "log_dir": "logs/rl/place_empty_cup",
    }
