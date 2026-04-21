#!/usr/bin/env python3
"""
VLA RL Fine-tuning — GRPO/PPO on RoboTwin

Runs closed-loop RL training with a VLA policy in the RoboTwin simulator.
Supports both GRPO (episode-level advantages) and PPO (step-level GAE).

Usage:
    # Mock mode (no simulator, CPU, for testing):
    python scripts/vla/rl/train_robotwin_grpo.py --mock --iterations 5

    # GRPO with RoboTwin simulator:
    python scripts/vla/rl/train_robotwin_grpo.py \
        --task place_empty_cup --num_envs 8 --iterations 100

    # PPO with value head:
    python scripts/vla/rl/train_robotwin_grpo.py \
        --algo ppo --task place_empty_cup --num_envs 8

    # Resume from pretrained checkpoint:
    python scripts/vla/rl/train_robotwin_grpo.py \
        --checkpoint checkpoints/vla_pretrain/checkpoint_final.pt \
        --task place_empty_cup

    # With real Qwen2-VL backbone:
    python scripts/vla/rl/train_robotwin_grpo.py \
        --model qwen2vl --model_name Qwen/Qwen2-VL-2B-Instruct \
        --task place_empty_cup --num_envs 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/RoboRenForce"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/tasks/RRF_robotwin"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/RRF_models"))

import torch
import torch.nn as nn

from RoboRenForce.prototype.embodied import BasePolicy, ForwardType, EmbodiedEnv
from RoboRenForce.networks.vlm.vlm_backbone_base import VLMBackbone, VLMBackboneCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import RegressionActionHeadCfg
from RoboRenForce.components.actor.vla_actor import VLAActorCfg
from RoboRenForce.algorithms.vla_training.grpo import GRPOAlgorithmCfg
from RoboRenForce.algorithms.vla_training.ppo import PPOAlgorithmCfg
from RoboRenForce.runners.vla.rl.vla_grpo_runner import VLAGRPORunner, VLAGRPORunnerCfg
from RoboRenForce.runners.vla.rl.vla_ppo_runner import VLAPPORunner, VLAPPORunnerCfg
from RoboRenForce.utils.configclass import configclass


# ===== Mock components for testing without simulator ===== #

class MockVLM(VLMBackbone):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, cfg.output_dim),
        )
        self.output_dim = cfg.output_dim

    def forward(self, image, text=None, **kwargs):
        return self.encoder(image)


@configclass
class MockVLMCfg(VLMBackboneCfg):
    class_type: type = MockVLM
    model_name: str = "mock"
    output_dim: int = 64
    freeze: bool = False


class MockRoboTwinEnv(EmbodiedEnv):
    """Mock environment mimicking RoboTwin interface for testing."""

    def __init__(self, num_envs=4, action_dim=14, state_dim=14,
                 image_size=(64, 64), max_episode_length=50, device="cpu"):
        self.num_envs = num_envs
        self.num_actions = action_dim
        self.num_obs = state_dim
        self.max_episode_length = max_episode_length
        self.device = torch.device(device)
        self._image_size = image_size
        self._step_count = 0

    def get_observations(self):
        return self._make_obs(), {}

    def reset(self):
        self._step_count = 0
        return self._make_obs(), {}

    def step(self, actions):
        self._step_count += 1
        obs = self._make_obs()
        # Random sparse reward (10% success rate)
        success = torch.rand(self.num_envs, device=self.device) < 0.1
        rewards = 5.0 * success.float()
        dones = success | (self._step_count >= self.max_episode_length)
        extras = {"termination": success, "timeout": dones & ~success}
        return obs, rewards, dones, extras

    def _make_obs(self):
        H, W = self._image_size
        return {
            "main_images": torch.randint(0, 255, (self.num_envs, H, W, 3),
                                         dtype=torch.uint8, device=self.device),
            "states": torch.randn(self.num_envs, self.num_obs, device=self.device),
            "task_descriptions": ["place empty cup"] * self.num_envs,
        }


class MockRLPolicy(BasePolicy):
    """Lightweight policy for RL mock testing."""

    def __init__(self, obs_dim=64, state_dim=14, action_dim=14, image_size=(64, 64)):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, obs_dim),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(obs_dim + state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(obs_dim + state_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._action_dim = action_dim

    def _encode(self, obs):
        images = obs["main_images"]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        if images.dim() == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2).contiguous()
        features = self.image_encoder(images)
        states = obs.get("states", torch.zeros(images.shape[0], 14, device=images.device))
        return torch.cat([features, states], dim=-1)

    def predict_action(self, obs, **kwargs):
        with torch.no_grad():
            h = self._encode(obs)
            return self.policy_head(h)

    def get_value(self, obs):
        h = self._encode(obs)
        return self.value_head(h.detach()).squeeze(-1)

    def forward(self, forward_type=ForwardType.INFERENCE, **kwargs):
        if forward_type == ForwardType.INFERENCE:
            return {"actions": self.predict_action(kwargs["obs"])}
        elif forward_type == ForwardType.PPO:
            obs = kwargs["obs"]
            actions = kwargs.get("actions")
            h = self._encode(obs)
            pred_actions = self.policy_head(h)
            if actions is not None:
                diff = actions.unsqueeze(1) - pred_actions if actions.dim() < pred_actions.dim() else actions - pred_actions
                logprobs = -0.5 * (diff ** 2).sum(dim=-1)
                if logprobs.dim() > 1:
                    logprobs = logprobs.mean(dim=-1)
            else:
                logprobs = -0.5 * (pred_actions ** 2).sum(dim=-1)
                if logprobs.dim() > 1:
                    logprobs = logprobs.mean(dim=-1)
            values = self.value_head(h.detach()).squeeze(-1)
            return {"pred_actions": pred_actions, "logprobs": logprobs, "values": values}
        raise ValueError(f"Unsupported forward type: {forward_type}")


# ===== Build functions ===== #

def build_env(args):
    """Build RoboTwin or mock environment."""
    if args.mock:
        return MockRoboTwinEnv(
            num_envs=args.num_envs,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            image_size=tuple(args.image_size),
            max_episode_length=args.max_steps,
            device=args.device,
        )

    from RRF_robotwin_tasks.envs.robotwin_env import RoboTwinEnv, RoboTwinTaskConfig

    # Set PYTHONPATH and ASSETS_PATH for RoboTwin
    robotwin_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../../third_party/RoboTwin"
    ))
    if os.path.exists(robotwin_path) and robotwin_path not in sys.path:
        sys.path.insert(0, robotwin_path)
    if not os.environ.get("ASSETS_PATH"):
        os.environ["ASSETS_PATH"] = robotwin_path

    task_cfg = RoboTwinTaskConfig(
        task_name=args.task,
        planner_backend="mplib",
        embodiment=["piper", "piper", 0.6],
        step_lim=args.max_steps,
    )

    env_cfg = {
        "task_config": task_cfg,
        "image_size": tuple(args.image_size),
        "action_dim": args.action_dim,
        "state_dim": args.state_dim,
        "max_episode_steps": args.max_steps,
        "use_custom_reward": True,
        "use_rel_reward": True,
        "reward_coef": args.reward_coef,
        "center_crop": True,
    }

    return RoboTwinEnv(env_cfg, num_envs=args.num_envs, device=args.device)


def build_policy(args):
    """Build VLA policy for RL."""
    if args.mock:
        return MockRLPolicy(
            obs_dim=64,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            image_size=tuple(args.image_size),
        )

    if args.model == "qwen2vl":
        from RRF_models.qwen2vl import build_qwen2vl_policy
        from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicyCfg
        from RoboRenForce.networks.vlm.qwen2vl import Qwen2VLCfg

        cfg = Qwen2VLPolicyCfg(
            actor_cfg=VLAActorCfg(
                vlm_backbone_cfg=Qwen2VLCfg(
                    model_name=args.model_name,
                    freeze=args.freeze_vlm,
                    device_map_auto=False,
                ),
                freeze_vlm=args.freeze_vlm,
                fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
                action_head_cfg=RegressionActionHeadCfg(
                    action_dim=args.action_dim, action_horizon=1, hidden_dims=[256, 256],
                ),
                use_proprioception=True,
            ),
            use_value_head=(args.algo == "ppo"),
            proprio_dim=args.state_dim,
        )
        return build_qwen2vl_policy(cfg=cfg)

    elif args.model == "qwen3vl":
        from RRF_models.qwen3vl import build_qwen3vl_policy
        from RRF_models.qwen3vl.qwen3vl_policy import Qwen3VLPolicyCfg
        from RoboRenForce.networks.vlm.qwen3vl import Qwen3VLCfg

        cfg = Qwen3VLPolicyCfg(
            actor_cfg=VLAActorCfg(
                vlm_backbone_cfg=Qwen3VLCfg(
                    model_name=args.model_name or "Qwen/Qwen3-VL-2B-Instruct",
                    freeze=args.freeze_vlm,
                    device_map_auto=False,
                ),
                freeze_vlm=args.freeze_vlm,
                fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
                action_head_cfg=RegressionActionHeadCfg(
                    action_dim=args.action_dim, action_horizon=1, hidden_dims=[256, 256],
                ),
                use_proprioception=True,
            ),
            use_value_head=(args.algo == "ppo"),
            proprio_dim=args.state_dim,
        )
        return build_qwen3vl_policy(cfg=cfg)

    else:
        raise ValueError(f"Unknown model: {args.model}. Use 'qwen2vl' or 'qwen3vl'.")


def build_runner(args, env, policy):
    """Build GRPO or PPO runner."""
    if args.algo == "grpo":
        cfg = VLAGRPORunnerCfg(
            grpo_cfg=GRPOAlgorithmCfg(
                group_size=args.group_size,
                clip_ratio_low=0.2,
                clip_ratio_high=0.28,
                learning_rate=args.lr,
                update_epochs=args.update_epochs,
                kl_beta=args.kl_coef,
                reward_coef=1.0,
            ),
            log_interval=1,
            save_interval=args.save_interval,
            checkpoint_dir=args.checkpoint_dir,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
        )
        return VLAGRPORunner(
            cfg, env=env, policy=policy,
            device=args.device, log_dir=args.log_dir,
        )

    elif args.algo == "ppo":
        cfg = VLAPPORunnerCfg(
            ppo_cfg=PPOAlgorithmCfg(
                gamma=0.99,
                gae_lambda=0.95,
                clip_ratio_low=0.2,
                clip_ratio_high=0.28,
                value_loss_coef=0.5,
                learning_rate=args.lr,
                update_epochs=args.update_epochs,
                kl_beta=args.kl_coef,
            ),
            log_interval=1,
            save_interval=args.save_interval,
            checkpoint_dir=args.checkpoint_dir,
        )
        return VLAPPORunner(
            cfg, env=env, policy=policy,
            device=args.device, log_dir=args.log_dir,
        )

    raise ValueError(f"Unknown algo: {args.algo}")


def main():
    parser = argparse.ArgumentParser(description="VLA RL Training on RoboTwin")

    # Mode
    parser.add_argument("--mock", action="store_true", help="Mock env + mock policy for testing")
    parser.add_argument("--algo", type=str, default="grpo", choices=["grpo", "ppo"])

    # Environment
    parser.add_argument("--task", type=str, default="place_empty_cup")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--state_dim", type=int, default=14)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--reward_coef", type=float, default=5.0)

    # Model
    parser.add_argument("--model", type=str, default="qwen2vl", choices=["qwen2vl", "qwen3vl"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--freeze_vlm", action="store_true", default=True)
    parser.add_argument("--no_freeze_vlm", dest="freeze_vlm", action="store_false")

    # Training
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--group_size", type=int, default=8, help="GRPO group size")
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--kl_coef", type=float, default=0.01)

    # Checkpointing
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained checkpoint to load")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/rl/")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="logs/rl/")

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=20)

    # Device
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mock:
        args.image_size = [64, 64]

    print(f"=== VLA RL Training ===")
    print(f"  Algorithm: {args.algo.upper()}")
    print(f"  Task: {args.task}")
    print(f"  Envs: {args.num_envs}")
    print(f"  Device: {args.device}")
    print(f"  Mock: {args.mock}")
    if not args.mock:
        print(f"  Model: {args.model} ({args.model_name})")
    print()

    # Build components
    env = build_env(args)
    policy = build_policy(args)
    policy = policy.to(args.device)

    # Load pretrained checkpoint if provided
    if args.checkpoint:
        print(f"Loading pretrained checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        if "policy_state_dict" in ckpt:
            policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
        elif "vla_actor_state_dict" in ckpt:
            policy.load_state_dict(ckpt["vla_actor_state_dict"], strict=False)
        else:
            policy.load_state_dict(ckpt, strict=False)
        print("  Checkpoint loaded.")

    # Build runner
    runner = build_runner(args, env, policy)

    # Train
    runner.learn(num_iterations=args.iterations)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    eval_metrics = runner.evaluate(num_episodes=args.eval_episodes)
    print(f"  Mean return: {eval_metrics['mean_return']:.3f}")
    print(f"  Success rate: {eval_metrics['success_rate']:.2%}")

    # Cleanup
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
