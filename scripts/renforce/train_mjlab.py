"""Train an RL agent on MJLab (MuJoCo Warp) locomotion environments.

Usage:
    # Go1 flat terrain with PPO (default)
    python scripts/renforce/train_mjlab.py --task Mjlab-Velocity-Flat-Unitree-Go1

    # G1 humanoid rough terrain
    python scripts/renforce/train_mjlab.py --task Mjlab-Velocity-Rough-Unitree-G1 --num_envs 2048

    # Custom algorithm via RRF gym registry
    python scripts/renforce/train_mjlab.py --task RoboRenForce-MJLab-Go1Flat-PPO --num_envs 4096

    # Resume from checkpoint
    python scripts/renforce/train_mjlab.py --task Mjlab-Velocity-Flat-Unitree-Go1 --resume logs/RFRL/.../model_5000.pt
"""

import argparse
import os
import pickle
import sys
from datetime import datetime

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent on MJLab environments.")

    # Task & environment
    parser.add_argument("--task", type=str, required=True,
                        help="MJLab task ID (e.g. Mjlab-Velocity-Flat-Unitree-Go1) "
                             "or RRF gym ID (e.g. RoboRenForce-MJLab-Go1Flat-PPO)")
    parser.add_argument("--num_envs", type=int, default=4096,
                        help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Compute device.")

    # Training overrides
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Override max training iterations.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name for logging.")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Override experiment name.")

    # Wrapper
    parser.add_argument("--wrapper", type=str, default="dynamic",
                        choices=["base", "dynamic", "group"],
                        help="Wrapper type: base, dynamic (default), or group.")
    parser.add_argument("--num_eval_envs", type=int, default=None,
                        help="Number of eval envs (only for group wrapper).")

    # Logging
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb"],
                        help="Logger backend.")
    parser.add_argument("--log_project", type=str, default=None,
                        help="WandB project name.")

    return parser.parse_args()


def load_mjlab_configs(task_name: str):
    """Load env and agent configs from MJLab task registry or RRF gym registry."""
    # Check if this is a RRF gym-registered task
    if task_name.startswith("RoboRenForce-MJLab-"):
        import gymnasium as gym
        import RRF_mjlab_tasks  # triggers registration
        spec = gym.spec(task_name)
        mjlab_task_id = spec.kwargs.get("mjlab_task_id")
        agent_cfg = spec.kwargs.get("RoboRenForce_entry_point")
        return mjlab_task_id, agent_cfg

    # Otherwise treat as a direct MJLab task ID
    # Use default PPO config
    from RRF_mjlab_tasks.locomotion.agents_ppo import MJLabLocoPPOCfg, MJLabLocoG1PPOCfg
    if "G1" in task_name:
        agent_cfg = MJLabLocoG1PPOCfg()
    else:
        agent_cfg = MJLabLocoPPOCfg()
    agent_cfg = agent_cfg.replace(experiment_name=task_name.replace("-", "_"))
    return task_name, agent_cfg


def make_log_dir(agent_cfg, run_name=None):
    """Create log directory matching RRF convention."""
    log_root = os.path.join("logs", "RFRL", agent_cfg.experiment_name)
    log_root = os.path.abspath(log_root)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = run_name or getattr(agent_cfg, 'run_name', '')
    if name:
        ts += f"_{name}"
    log_dir = os.path.join(log_root, ts)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    return log_dir


def main():
    args = parse_args()
    print(f"[INFO] Task: {args.task}, Device: {args.device}, Envs: {args.num_envs}")

    # Load configs
    mjlab_task_id, agent_cfg = load_mjlab_configs(args.task)

    # Apply CLI overrides
    agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.logger:
        agent_cfg.logger_cfg.logger = args.logger
    if args.log_project and hasattr(agent_cfg.logger_cfg, 'wandb_project'):
        agent_cfg.logger_cfg.wandb_project = args.log_project

    # Create environment with wrapper
    from RRF_mjlab_tasks.mjlab_utils.gym_config import make_mjlab_env
    env = make_mjlab_env(
        task_name=mjlab_task_id,
        num_envs=args.num_envs,
        device=args.device,
        wrapper=args.wrapper,
    )
    print(f"[INFO] Env created: num_envs={env.num_envs}, obs={env.num_obs}, "
          f"actions={env.num_actions}, privileged_obs={env.num_privileged_obs}")

    # Create log directory
    log_dir = make_log_dir(agent_cfg, args.run_name)
    print(f"[INFO] Logging to: {log_dir}")

    # Build runner from agent config
    from RoboRenForce.runners import BaseRunner
    runner: BaseRunner = agent_cfg.construct_from_cfg(env, log_dir, device=args.device)
    print(f"[INFO] Runner: {type(runner).__name__}")
    print(f"[INFO] Algorithm: {runner.alg}")

    # Resume from checkpoint if specified
    if args.resume:
        resume_path = os.path.abspath(args.resume)
        print(f"[INFO] Resuming from: {resume_path}")
        runner.load(resume_path)

    # Save configs for reproducibility
    try:
        with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
            pickle.dump(agent_cfg, f)
        with open(os.path.join(log_dir, "params", "args.pkl"), "wb") as f:
            pickle.dump(args, f)
    except Exception as e:
        print(f"[WARN] Could not save configs: {e}")

    # Train
    max_iter = args.max_iterations or agent_cfg.max_iterations
    print(f"[INFO] Starting training for {max_iter} iterations...")
    runner.learn(num_learning_iterations=max_iter)

    env.close()
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
