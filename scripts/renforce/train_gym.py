"""
Train classic Gymnasium environments with RoboRenForce.

Usage (registered env IDs):
    python scripts/renforce/train_gym.py --task RoboRenForce-Gym-HalfCheetah-SAC
    python scripts/renforce/train_gym.py --task RoboRenForce-Gym-Walker2d-PPO
    python scripts/renforce/train_gym.py --task RoboRenForce-Gym-Ant-DSACT

Usage (raw gymnasium IDs — uses default DSACT config):
    python scripts/renforce/train_gym.py --task HalfCheetah-v4
    python scripts/renforce/train_gym.py --task Pendulum-v1
"""

import argparse
import os
import gymnasium as gym
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from RoboRenForce.utils import argtool
from RoboRenForce.runners import BaseRunner

# Register RRF gym tasks
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/tasks/RRF_gym"))
import RRF_gym_tasks  # noqa: triggers registration


def main():
    parser = argparse.ArgumentParser(description="Train on Gymnasium environments")
    parser.add_argument("--task", type=str, required=True,
                        help="RoboRenForce env ID (e.g. RoboRenForce-Gym-HalfCheetah-SAC) "
                             "or raw gym ID (e.g. HalfCheetah-v4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    argtool.add_args_group(parser)
    args = parser.parse_args()

    task = args.task

    if task.startswith("RoboRenForce-Gym-"):
        # Registered env — extract config from registry
        spec = gym.spec(task)
        gym_task_id = spec.kwargs["gym_task_id"]
        agent_cfg = spec.kwargs["RoboRenForce_entry_point"]
    else:
        # Raw gymnasium ID — use default DSACT config
        from RRF_gym_tasks.locomotion.agents import GymDSACTCfg
        gym_task_id = task
        agent_cfg = GymDSACTCfg()
        agent_cfg.experiment_name = task

    agent_cfg.seed = args.seed
    log_dir = argtool.make_log_dir(agent_cfg)

    # Create env
    from RRF_gym_tasks.envs import GymVecEnv
    env = GymVecEnv(task_name=gym_task_id, device=args.device, seed=args.seed)

    # Create runner
    runner: BaseRunner = agent_cfg.construct_from_cfg(
        env=env,
        log_dir=log_dir,
        device=args.device,
    )
    if agent_cfg.resume:
        runner.load(agent_cfg.load_checkpoint)

    print(f"Task:      {gym_task_id}")
    print(f"Algorithm: {agent_cfg.run_name}")
    print(f"Device:    {args.device}")
    print(f"Log dir:   {log_dir}")
    print()

    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
    env.close()


if __name__ == "__main__":
    main()
