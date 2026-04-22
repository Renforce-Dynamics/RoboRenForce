"""
Config: LIBERO-10 — Supervised Pretraining with VLA

Reference config for pretraining a VLA on the LIBERO-10 benchmark (10 tasks).

Usage:
    python -m RoboRenForce.runners.vla.pretrain.run --config \\
        source/tasks/RRF_libero/RRF_libero_tasks/configs/libero_10_pretrain.py
"""


def get_config():
    return {
        "env": {
            "env_class": "RRF_libero_tasks.envs.LiberoRRFEnv",
            "task_suite_name": "libero_10",
            "image_size": (224, 224),
            "action_dim": 7,
            "state_dim": 7,
            "max_episode_steps": 300,
            "has_wrist_camera": True,
            "num_envs": 8,
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
            "num_eval_episodes": 20,
            "success_metric": "success_once",
        },
    }
