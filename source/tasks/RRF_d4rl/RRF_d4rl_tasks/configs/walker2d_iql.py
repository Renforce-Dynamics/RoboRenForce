"""
Config: D4RL Walker2d-Medium — Offline RL with IQL

Reference config for Implicit Q-Learning on walker2d-medium-v2.

Usage:
    python scripts/renforce/train_gym.py --task walker2d-medium-v2 \\
        --algo iql --max_iterations 1000000
"""


def get_config():
    return {
        "env": {
            "env_class": "RRF_d4rl_tasks.envs.D4RLRRFEnv",
            "task_name": "walker2d-medium-v2",
            "state_dim": 17,
            "action_dim": 6,
            "max_episode_steps": 1000,
            "num_envs": 1,
        },
        "training": {
            "algorithm": "iql",
            "batch_size": 256,
            "num_steps": 1_000_000,
            "learning_rate": 3e-4,
            "discount": 0.99,
            "tau": 0.005,
            "expectile": 0.7,
            "temperature": 3.0,
        },
        "eval": {
            "eval_interval": 10_000,
            "num_eval_episodes": 10,
            "normalize_score": True,
        },
    }
