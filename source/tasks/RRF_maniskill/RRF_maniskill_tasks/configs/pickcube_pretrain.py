"""
Config: ManiSkill PickCube — Supervised Pretraining with VLA

Reference config for pretraining on PickCube-v1 (Franka robot, single object).

Usage:
    python -m RoboRenForce.runners.vla.pretrain.run --config \\
        source/tasks/RRF_maniskill/RRF_maniskill_tasks/configs/pickcube_pretrain.py
"""


def get_config():
    return {
        "env": {
            "env_class": "RRF_maniskill_tasks.envs.ManiSkillRRFEnv",
            "task_name": "PickCube-v1",
            "image_size": (224, 224),
            "action_dim": 7,
            "state_dim": 25,
            "max_episode_steps": 200,
            "obs_mode": "rgbd",
            "control_mode": "pd_ee_delta_pose",
            "num_envs": 16,
        },
        "training": {
            "model_type": "qwen2vl",
            "batch_size": 32,
            "num_epochs": 50,
            "learning_rate": 1e-4,
            "warmup_steps": 500,
            "action_loss_type": "l1",
        },
        "eval": {
            "eval_interval": 5,
            "num_eval_episodes": 50,
            "success_metric": "success",
        },
    }
