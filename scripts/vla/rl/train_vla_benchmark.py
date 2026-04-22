#!/usr/bin/env python3
"""
VLA Training on Benchmarks — LIBERO, ManiSkill, CALVIN, D4RL

Unified training script for all RLinf-bridged benchmark environments.
Supports VLA pretraining (offline) and RL fine-tuning (online).

Usage:
    # LIBERO-10 with mock VLA (testing):
    python scripts/vla/rl/train_vla_benchmark.py \
        --benchmark libero --task_suite libero_10 --mock --num_envs 4

    # ManiSkill PickCube:
    python scripts/vla/rl/train_vla_benchmark.py \
        --benchmark maniskill --task_name PickCube-v1 --num_envs 16

    # CALVIN ABCD:
    python scripts/vla/rl/train_vla_benchmark.py \
        --benchmark calvin --task_suite calvin_abcd --num_envs 4

    # D4RL offline RL:
    python scripts/vla/rl/train_vla_benchmark.py \
        --benchmark d4rl --task_name walker2d-medium-v2

    # With real Qwen2-VL backbone:
    python scripts/vla/rl/train_vla_benchmark.py \
        --benchmark libero --task_suite libero_10 --num_envs 8 \
        --model qwen2vl --model_name Qwen/Qwen2-VL-2B-Instruct

    # Resume from checkpoint:
    python scripts/vla/rl/train_vla_benchmark.py \
        --benchmark libero --task_suite libero_10 \
        --checkpoint checkpoints/libero_pretrain/best.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/RoboRenForce"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/tasks/RRF_libero"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/tasks/RRF_maniskill"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/tasks/RRF_calvin"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/tasks/RRF_d4rl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/RRF_models"))

import torch
import torch.nn as nn

from RoboRenForce.prototype.embodied import BasePolicy, ForwardType, EmbodiedEnv


# ===== Mock components for testing without simulators ===== #

class MockBenchmarkEnv(EmbodiedEnv):
    """Mock environment for any benchmark — testing without simulator."""

    def __init__(self, num_envs=4, action_dim=7, state_dim=7,
                 image_size=(64, 64), has_wrist=False,
                 max_episode_length=100, device="cpu"):
        self.num_envs = num_envs
        self.num_actions = action_dim
        self.num_obs = state_dim
        self.max_episode_length = max_episode_length
        self.device = torch.device(device)
        self._image_size = image_size
        self._has_wrist = has_wrist
        self._step_count = 0

    def get_observations(self):
        return self._make_obs(), {}

    def reset(self):
        self._step_count = 0
        return self._make_obs(), {}

    def step(self, actions):
        self._step_count += 1
        obs = self._make_obs()
        success = torch.rand(self.num_envs, device=self.device) < 0.05
        rewards = success.float()
        dones = success | (self._step_count >= self.max_episode_length)
        extras = {"termination": success, "timeout": dones & ~success}
        return obs, rewards, dones, extras

    def _make_obs(self):
        H, W = self._image_size
        obs = {
            "main_images": torch.randint(0, 255, (self.num_envs, H, W, 3),
                                          dtype=torch.uint8, device=self.device),
            "states": torch.randn(self.num_envs, self.num_obs, device=self.device),
            "task_descriptions": ["mock task"] * self.num_envs,
        }
        if self._has_wrist:
            obs["wrist_images"] = torch.randint(
                0, 255, (self.num_envs, H, W, 3),
                dtype=torch.uint8, device=self.device)
        return obs


class MockPolicy(BasePolicy):
    """Lightweight mock policy for testing."""

    def __init__(self, state_dim=7, action_dim=7, image_size=(64, 64)):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, 64),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(64 + state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(64 + state_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _encode(self, obs):
        images = obs["main_images"]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        if images.dim() == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2).contiguous()
        features = self.image_encoder(images)
        states = obs.get("states", torch.zeros(images.shape[0], 7, device=images.device))
        return torch.cat([features, states], dim=-1)

    def predict_action(self, obs, **kwargs):
        with torch.no_grad():
            return self.policy_head(self._encode(obs))

    def get_value(self, obs):
        return self.value_head(self._encode(obs).detach()).squeeze(-1)

    def forward(self, forward_type=ForwardType.INFERENCE, **kwargs):
        obs = kwargs.get("obs", {})
        if forward_type == ForwardType.INFERENCE:
            return {"actions": self.predict_action(obs)}
        h = self._encode(obs)
        pred = self.policy_head(h)
        actions = kwargs.get("actions")
        if actions is not None:
            diff = actions - pred
            logprobs = -0.5 * (diff ** 2).sum(dim=-1)
        else:
            logprobs = -0.5 * (pred ** 2).sum(dim=-1)
        values = self.value_head(h.detach()).squeeze(-1)
        return {"pred_actions": pred, "logprobs": logprobs, "values": values}


# ===== Environment builders ===== #

BENCHMARK_DEFAULTS = {
    "libero": {"action_dim": 7, "state_dim": 7, "max_steps": 300, "has_wrist": True},
    "maniskill": {"action_dim": 7, "state_dim": 25, "max_steps": 200, "has_wrist": False},
    "calvin": {"action_dim": 7, "state_dim": 7, "max_steps": 360, "has_wrist": True},
    "d4rl": {"action_dim": 6, "state_dim": 17, "max_steps": 1000, "has_wrist": False},
}


def build_env(args):
    """Build benchmark environment (mock or real)."""
    defaults = BENCHMARK_DEFAULTS.get(args.benchmark, BENCHMARK_DEFAULTS["libero"])

    if args.mock:
        return MockBenchmarkEnv(
            num_envs=args.num_envs,
            action_dim=args.action_dim or defaults["action_dim"],
            state_dim=args.state_dim or defaults["state_dim"],
            image_size=tuple(args.image_size),
            has_wrist=defaults["has_wrist"],
            max_episode_length=args.max_steps or defaults["max_steps"],
            device=args.device,
        )

    action_dim = args.action_dim or defaults["action_dim"]
    state_dim = args.state_dim or defaults["state_dim"]
    max_steps = args.max_steps or defaults["max_steps"]

    if args.benchmark == "libero":
        from RRF_libero_tasks.envs.libero_env import LiberoRRFEnv
        cfg = {
            "task_suite_name": args.task_suite or "libero_10",
            "image_size": tuple(args.image_size),
            "action_dim": action_dim,
            "state_dim": state_dim,
            "max_episode_steps": max_steps,
            "has_wrist_camera": True,
            "seed": args.seed,
        }
        return LiberoRRFEnv(cfg, num_envs=args.num_envs, device=args.device)

    elif args.benchmark == "maniskill":
        from RRF_maniskill_tasks.envs.maniskill_env import ManiSkillRRFEnv
        cfg = {
            "task_name": args.task_name or "PickCube-v1",
            "image_size": tuple(args.image_size),
            "action_dim": action_dim,
            "state_dim": state_dim,
            "max_episode_steps": max_steps,
            "obs_mode": "rgbd",
            "control_mode": "pd_ee_delta_pose",
            "seed": args.seed,
        }
        return ManiSkillRRFEnv(cfg, num_envs=args.num_envs, device=args.device)

    elif args.benchmark == "calvin":
        from RRF_calvin_tasks.envs.calvin_env import CalvinRRFEnv
        cfg = {
            "task_suite_name": args.task_suite or "calvin_abcd",
            "image_size": tuple(args.image_size),
            "action_dim": action_dim,
            "state_dim": state_dim,
            "max_episode_steps": max_steps,
            "has_wrist_camera": True,
            "seed": args.seed,
        }
        return CalvinRRFEnv(cfg, num_envs=args.num_envs, device=args.device)

    elif args.benchmark == "d4rl":
        from RRF_d4rl_tasks.envs.d4rl_env import D4RLRRFEnv
        cfg = {
            "task_name": args.task_name or "walker2d-medium-v2",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_episode_steps": max_steps,
            "seed": args.seed,
        }
        return D4RLRRFEnv(cfg, num_envs=args.num_envs, device=args.device)

    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


def build_policy(args):
    """Build VLA or mock policy."""
    defaults = BENCHMARK_DEFAULTS.get(args.benchmark, BENCHMARK_DEFAULTS["libero"])
    action_dim = args.action_dim or defaults["action_dim"]
    state_dim = args.state_dim or defaults["state_dim"]

    if args.mock or args.model == "mock":
        return MockPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            image_size=tuple(args.image_size),
        )

    if args.model == "qwen2vl":
        from RRF_models.qwen2vl import build_qwen2vl_policy
        return build_qwen2vl_policy(
            model_name=args.model_name,
            action_dim=action_dim,
            state_dim=state_dim,
            freeze_vlm=args.freeze_vlm,
        )

    raise ValueError(f"Unknown model: {args.model}")


def run_eval_loop(env, policy, args):
    """Simple evaluation loop: rollout and report success rate."""
    device = torch.device(args.device)
    policy = policy.to(device)
    policy.eval()

    num_episodes = args.eval_episodes
    total_success = 0
    total_return = 0.0
    episodes_done = 0

    obs, _ = env.reset()
    ep_returns = torch.zeros(env.num_envs, device=device)

    while episodes_done < num_episodes:
        with torch.no_grad():
            out = policy(forward_type=ForwardType.INFERENCE, obs=obs)
            actions = out["actions"]

        obs, rewards, dones, extras = env.step(actions)
        ep_returns += rewards

        for i in range(env.num_envs):
            if dones[i]:
                episodes_done += 1
                total_return += ep_returns[i].item()
                if extras.get("termination") is not None and extras["termination"][i]:
                    total_success += 1
                ep_returns[i] = 0.0
                if episodes_done >= num_episodes:
                    break

    success_rate = total_success / num_episodes * 100
    avg_return = total_return / num_episodes
    print(f"\n{'='*50}")
    print(f"Eval results ({args.benchmark} / {args.task_suite or args.task_name}):")
    print(f"  Episodes:     {num_episodes}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Avg return:   {avg_return:.3f}")
    print(f"{'='*50}")


def run_rl_loop(env, policy, args):
    """Simple RL training loop with GRPO/PPO."""
    from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithmCfg
    from RoboRenForce.runners.vla.rl.vla_grpo_runner import VLAGRPORunner, VLAGRPORunnerCfg
    from RoboRenForce.utils.configclass import configclass

    device = torch.device(args.device)
    policy = policy.to(device)

    runner_cfg = VLAGRPORunnerCfg(
        max_iterations=args.iterations,
        num_steps_per_env=args.rollout_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        experiment_name=f"{args.benchmark}_{args.task_suite or args.task_name}",
        algorithm_cfg=GRPOAlgorithmCfg(
            learning_rate=args.lr,
            num_update_epochs=args.update_epochs,
        ),
    )

    runner = VLAGRPORunner(env, policy, runner_cfg, log_dir=args.log_dir, device=device)
    runner.learn()


def main():
    parser = argparse.ArgumentParser(description="VLA Benchmark Training")

    # Benchmark selection
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["libero", "maniskill", "calvin", "d4rl"],
                        help="Which benchmark to run")
    parser.add_argument("--task_suite", type=str, default=None,
                        help="Task suite name (for LIBERO: libero_10/90, CALVIN: calvin_abcd)")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Specific task name (for ManiSkill: PickCube-v1, D4RL: walker2d-medium-v2)")

    # Environment
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--state_dim", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--model", type=str, default="mock", choices=["mock", "qwen2vl"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--freeze_vlm", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mock", action="store_true", help="Use mock env + policy for testing")

    # Training
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "rl"],
                        help="Training mode: eval (rollout only) or rl (GRPO/PPO)")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rollout_steps", type=int, default=16)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_episodes", type=int, default=50)

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    print(f"Benchmark: {args.benchmark}")
    print(f"Task:      {args.task_suite or args.task_name or 'default'}")
    print(f"Mode:      {args.mode}")
    print(f"Mock:      {args.mock}")
    print(f"Device:    {args.device}")
    print()

    env = build_env(args)
    policy = build_policy(args)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        if "policy_state_dict" in ckpt:
            policy.load_state_dict(ckpt["policy_state_dict"])
        elif "model_state_dict" in ckpt:
            policy.load_state_dict(ckpt["model_state_dict"])
        else:
            policy.load_state_dict(ckpt)

    if args.mode == "eval":
        run_eval_loop(env, policy, args)
    elif args.mode == "rl":
        run_rl_loop(env, policy, args)

    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
