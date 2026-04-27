"""Shared CLI parsing + runner construction for VLA RL training scripts.

Each ``train_<benchmark>.py`` calls into here. This module MUST NOT import
any ``RRF_<benchmark>_vla_rl_tasks`` package — the per-benchmark script does
that import to register the task IDs, then hands ``args`` here.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import gymnasium as gym


def make_parser(benchmark_name: str) -> argparse.ArgumentParser:
    """Standard CLI for any VLA RL benchmark."""
    p = argparse.ArgumentParser(description=f"VLA RL training on {benchmark_name}")
    p.add_argument("--task", required=True,
                   help=f"Registered {benchmark_name} VLA-RL task ID "
                        "(e.g. RoboTwin-PlaceCup-GRPO-v0)")
    p.add_argument("--num_envs", type=int, default=None,
                   help="Override env_cfg.num_envs.")
    p.add_argument("--device", default=None,
                   help="Compute device (defaults to cuda if available).")
    p.add_argument("--max_iterations", type=int, default=None,
                   help="Override runner_cfg max iterations.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed.")
    p.add_argument("--logdir", default=None,
                   help="Override log directory (default: logs/RFRL/<task>/<ts>).")
    p.add_argument("--checkpoint", default=None,
                   help="Optional pretrained checkpoint to load into the policy.")
    return p


def _resolve_device(arg_device: str | None) -> str:
    if arg_device is not None:
        return arg_device
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_log_dir(task_id: str, override: str | None) -> str:
    if override:
        os.makedirs(override, exist_ok=True)
        return override
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join("logs", "RFRL", task_id, ts))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def run(args: argparse.Namespace) -> int:
    """Resolve task spec → build env+policy+runner → ``runner.learn()``.

    Returns process exit code (0 on success).
    """
    spec = gym.spec(args.task)
    env_cfg    = spec.kwargs["env_cfg_entry_point"]
    runner_cfg = spec.kwargs["RoboRenForce_entry_point"]

    device = _resolve_device(args.device)
    log_dir = _make_log_dir(args.task, args.logdir)

    # Apply CLI overrides BEFORE building the env (env constructor reads num_envs).
    if args.num_envs is not None:
        env_cfg.num_envs = args.num_envs
    env_cfg.device = device

    if args.max_iterations is not None:
        # Both VLAGRPORunner and VLAPPORunner take num_iterations as a learn() arg,
        # but we also stash it here so the script can pass it through.
        runner_cfg._cli_max_iterations = args.max_iterations
    if args.seed is not None:
        # Seed is set by the env / policy at build time. Stash for the runner cfg.
        runner_cfg._cli_seed = args.seed

    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Num envs: {env_cfg.num_envs}")
    print(f"[INFO] Log dir: {log_dir}")

    # Build env
    env = env_cfg.build()
    print(f"[INFO] Env built: {type(env).__name__} (num_envs={env.num_envs})")

    # Build policy via the cfg's build_policy callable
    if not hasattr(runner_cfg, "build_policy"):
        raise AttributeError(
            f"Runner cfg {type(runner_cfg).__name__} is missing the "
            "`build_policy` static-method attribute (per task convention)."
        )
    policy = runner_cfg.build_policy().to(device)
    print(f"[INFO] Policy built: {type(policy).__name__}")

    # Optional checkpoint
    if args.checkpoint:
        import torch
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = (
            ckpt.get("policy_state_dict")
            or ckpt.get("vla_actor_state_dict")
            or ckpt
        )
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        print(f"[INFO] Loaded checkpoint: {args.checkpoint} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # Build runner
    runner_cls = type(runner_cfg).__mro__[1]  # parent class is the actual VLA*RunnerCfg
    # Use class_type from the cfg to avoid relying on MRO
    runner = runner_cfg.class_type(
        cfg=runner_cfg, env=env, policy=policy, device=device, log_dir=log_dir,
    )
    print(f"[INFO] Runner: {type(runner).__name__}")

    num_iter = args.max_iterations or 100
    print(f"[INFO] Starting training for {num_iter} iterations...")
    runner.learn(num_iterations=num_iter)

    if hasattr(env, "close"):
        env.close()
    print("[INFO] Training complete.")
    return 0
