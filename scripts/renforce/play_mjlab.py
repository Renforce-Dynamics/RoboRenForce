"""Evaluate a trained RL agent on MJLab environments.

Usage:
    # Play with a trained checkpoint
    python scripts/renforce/play_mjlab.py --target logs/RFRL/.../model_5000.pt

    # Specify task and num_envs
    python scripts/renforce/play_mjlab.py --target logs/RFRL/.../model_5000.pt \
        --task Mjlab-Velocity-Flat-Unitree-Go1 --num_envs 64 --steps 1000
"""

import argparse
import os
import pickle

import torch
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Play/evaluate RL agent on MJLab environments.")

    parser.add_argument("--target", type=str, required=True,
                        help="Path to model checkpoint.")
    parser.add_argument("--task", type=str, default=None,
                        help="MJLab task ID. If None, loaded from checkpoint params.")
    parser.add_argument("--num_envs", type=int, default=64,
                        help="Number of environments.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Compute device.")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of evaluation steps (0 = infinite).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed.")

    return parser.parse_args()


def main():
    args = parse_args()
    resume_path = os.path.abspath(args.target)
    run_dir = os.path.dirname(resume_path)

    # Try to load saved configs
    task_name = args.task
    agent_cfg = None

    params_dir = os.path.join(run_dir, "params")
    if os.path.exists(os.path.join(params_dir, "args.pkl")):
        with open(os.path.join(params_dir, "args.pkl"), "rb") as f:
            saved_args = pickle.load(f)
        if task_name is None:
            task_name = saved_args.task
    if os.path.exists(os.path.join(params_dir, "agent.pkl")):
        with open(os.path.join(params_dir, "agent.pkl"), "rb") as f:
            agent_cfg = pickle.load(f)

    if task_name is None:
        raise ValueError("No --task specified and could not find saved args. "
                         "Please provide --task explicitly.")

    # Load configs if not from pickle
    if agent_cfg is None:
        from scripts.renforce.train_mjlab import load_mjlab_configs
        _, agent_cfg = load_mjlab_configs(task_name)

    # Resolve mjlab task ID
    if task_name.startswith("RoboRenForce-MJLab-"):
        import gymnasium as gym
        import RRF_mjlab_tasks
        spec = gym.spec(task_name)
        mjlab_task_id = spec.kwargs.get("mjlab_task_id")
    else:
        mjlab_task_id = task_name

    # Create environment (play mode uses fewer envs)
    from RRF_mjlab_tasks.mjlab_utils.gym_config import make_mjlab_env
    env = make_mjlab_env(
        task_name=mjlab_task_id,
        num_envs=args.num_envs,
        device=args.device,
        play=True,
        wrapper="dynamic",
    )
    print(f"[INFO] Env: num_envs={env.num_envs}, obs={env.num_obs}, actions={env.num_actions}")

    # Build runner and load checkpoint
    from RoboRenForce.runners import BaseRunner
    runner: BaseRunner = agent_cfg.construct_from_cfg(env, run_dir, device=args.device)
    runner.load(resume_path, load_optimizer=False)
    runner.eval_mode()
    print(f"[INFO] Loaded checkpoint: {resume_path}")

    policy = runner.get_inference_policy(device=args.device)

    # Run evaluation loop
    obs, _ = env.get_observations()
    total_reward = 0.0
    ep_count = 0

    pbar = tqdm.tqdm(range(args.steps)) if args.steps > 0 else tqdm.tqdm()
    try:
        for step in pbar:
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, infos = env.step(actions)

            total_reward += rewards.mean().item()
            ep_count += dones.sum().item()

            if args.steps > 0:
                pbar.set_postfix(
                    avg_rew=f"{total_reward / (step + 1):.4f}",
                    episodes=int(ep_count)
                )
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted.")

    steps_done = args.steps if args.steps > 0 else step + 1
    print(f"\n[INFO] Evaluation complete: {steps_done} steps, "
          f"{int(ep_count)} episodes, avg_reward={total_reward / max(1, steps_done):.4f}")

    env.close()


if __name__ == "__main__":
    main()
