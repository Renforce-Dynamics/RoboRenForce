"""Train an AMP agent on MJLab velocity-tracking environments.

Mirrors :mod:`scripts.renforce.train_mjlab` but pre-wraps the env with
:class:`MJLabAMPEnvWrapper` (the AMP runner expects the seven-tuple step API)
and lets the user point at one or more motion-clip ``.npz`` files.

Usage:
    python scripts/renforce/train_mjlab_amp.py \\
        --task RoboRenForce-MJLab-AMP-Velocity-Flat-G1 \\
        --motion_files data/demo/g1/walk.npz data/demo/g1/jog.npz \\
        --num_envs 4096
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
    parser = argparse.ArgumentParser(description="Train AMP agent on MJLab.")
    parser.add_argument("--task", type=str, required=True,
                        help="RRF gym ID (e.g. RoboRenForce-MJLab-AMP-Velocity-Flat-G1).")
    parser.add_argument("--motion_files", type=str, nargs="+", required=True,
                        help="Paths to motion-clip .npz files used as expert data.")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--amp_reward_coef", type=float, default=None,
                        help="Override AMPPPO amp_reward_coef, e.g. 0.25 for half of the default 0.5.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb"])
    parser.add_argument("--log_project", type=str, default=None)
    return parser.parse_args()


def load_amp_configs(task_name: str):
    """Resolve env cfg factory + agent cfg from the RRF gym registry."""
    import gymnasium as gym
    import RRF_mjlab_tasks  # triggers registration  # noqa: F401

    spec = gym.spec(task_name)
    factory_path = spec.kwargs.get("RRF_amp_env_cfg_factory")
    agent_cfg = spec.kwargs.get("RoboRenForce_entry_point")
    if factory_path is None or agent_cfg is None:
        raise ValueError(f"Task {task_name!r} is not an AMP RRF task.")

    module_path, func_name = factory_path.split(":")
    import importlib
    factory = getattr(importlib.import_module(module_path), func_name)
    return factory, agent_cfg


def make_log_dir(agent_cfg, run_name=None):
    log_root = os.path.join("logs", "RFRL", agent_cfg.experiment_name)
    log_root = os.path.abspath(log_root)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = run_name or getattr(agent_cfg, "run_name", "")
    if name:
        ts += f"_{name}"
    log_dir = os.path.join(log_root, ts)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    return log_dir


def main():
    args = parse_args()
    print(f"[INFO] Task: {args.task}, Device: {args.device}, Envs: {args.num_envs}")

    factory, agent_cfg = load_amp_configs(args.task)

    # CLI overrides
    agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.amp_reward_coef is not None:
        agent_cfg.algorithm.amp_reward_coef = args.amp_reward_coef
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.logger:
        agent_cfg.logger_cfg.logger = args.logger
    if args.log_project and hasattr(agent_cfg.logger_cfg, "wandb_project"):
        agent_cfg.logger_cfg.wandb_project = args.log_project

    # Motion files: required at run time (Option A — curated paths).
    for f in args.motion_files:
        if not os.path.isfile(f):
            sys.exit(f"[ERR] Motion file not found: {f}")
    agent_cfg.amp_data.motion_files = list(args.motion_files)

    # Build env
    env_cfg = factory(play=False)
    env_cfg.scene.num_envs = args.num_envs

    from mjlab.envs import ManagerBasedRlEnv
    from RRF_mjlab_tasks.mjlab_utils.amp_env_wrapper import MJLabAMPEnvWrapper

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = MJLabAMPEnvWrapper(env)
    print(f"[INFO] Env created: num_envs={env.num_envs}, obs={env.num_obs}, "
          f"actions={env.num_actions}, privileged_obs={env.num_privileged_obs}")

    log_dir = make_log_dir(agent_cfg, args.run_name)
    print(f"[INFO] Logging to: {log_dir}")

    # Build runner
    from RoboRenForce.runners import BaseRunner
    runner: BaseRunner = agent_cfg.construct_from_cfg(env, log_dir, device=args.device)
    print(f"[INFO] Runner: {type(runner).__name__}")

    if args.resume:
        resume_path = os.path.abspath(args.resume)
        print(f"[INFO] Resuming from: {resume_path}")
        runner.load(resume_path)

    try:
        with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
            pickle.dump(agent_cfg, f)
        with open(os.path.join(log_dir, "params", "args.pkl"), "wb") as f:
            pickle.dump(args, f)
    except Exception as e:
        print(f"[WARN] Could not save configs: {e}")

    max_iter = args.max_iterations or agent_cfg.max_iterations
    print(f"[INFO] Starting AMP training for {max_iter} iterations...")
    runner.learn(num_learning_iterations=max_iter)

    env.close()
    print("[INFO] AMP training complete.")


if __name__ == "__main__":
    main()
