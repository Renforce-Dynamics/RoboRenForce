"""
Config: CALVIN ABCD — Supervised Pretraining with VLA

Reference config for pretraining on CALVIN with all four scenes (A, B, C, D).
CALVIN evaluates long-horizon manipulation via 5-subtask chained sequences.

Usage:
    python -m RoboRenForce.runners.vla.pretrain.run --config \\
        source/tasks/RRF_calvin/RRF_calvin_tasks/configs/calvin_abcd_pretrain.py
"""


def get_config():
    return {
        "env": {
            "env_class": "RRF_calvin_tasks.envs.CalvinRRFEnv",
            "task_suite_name": "calvin_abcd",
            "image_size": (224, 224),
            "action_dim": 7,
            "state_dim": 7,
            "max_episode_steps": 360,
            "has_wrist_camera": True,
            "num_envs": 4,
        },
        "training": {
            "model_type": "qwen2vl",
            "batch_size": 16,
            "num_epochs": 50,
            "learning_rate": 1e-4,
            "warmup_steps": 500,
            "action_loss_type": "l1",
            "action_chunk_size": 1,
        },
        "eval": {
            "eval_interval": 5,
            "num_eval_episodes": 100,
            "success_metric": "avg_subtask_completed",
            "num_subtasks": 5,
        },
    }
