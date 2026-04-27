"""
LIBERO Environment Wrapper for RoboRenForce

Directly wraps the LIBERO simulator (robosuite-based) behind the EmbodiedEnv interface.
No dependency on RLinf — uses upstream libero package directly.

LIBERO task suites:
    - libero_10:       10 manipulation tasks (standard benchmark)
    - libero_spatial:  90 tasks testing spatial reasoning
    - libero_object:   90 tasks testing object generalization
    - libero_goal:     90 tasks testing goal-driven behavior
    - libero_90:       90-task aggregate

Observation format:
    main_images:      [B, H, W, 3] - front camera RGB (agentview)
    wrist_images:     [B, H, W, 3] - wrist camera RGB (eye_in_hand)
    states:           [B, 7]       - EEF pos(3) + axisangle(3) + gripper(1)
    task_descriptions: list[str]   - language instructions

Action format: [B, 7] — 6D EEF pose delta + 1D gripper

Dependencies:
    pip install robosuite
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO
    pip install -e LIBERO
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from RoboRenForce.prototype.embodied import EmbodiedEnv


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert (x, y, z, w) quaternion to axis-angle (3D)."""
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(w) / den).astype(np.float32)


def _rotate180(img: np.ndarray) -> np.ndarray:
    """Rotate image 180 degrees (LIBERO camera convention)."""
    return img[::-1, ::-1]


def _extract_obs(raw_obs: dict, collect_wrist: bool = True) -> dict:
    """Extract standardized obs from a single LIBERO env observation."""
    main_img = _rotate180(raw_obs["agentview_image"])
    state = np.concatenate([
        raw_obs["robot0_eef_pos"].astype(np.float32),
        _quat2axisangle(raw_obs["robot0_eef_quat"]),
        raw_obs["robot0_gripper_qpos"].astype(np.float32).flatten()[:1],
    ])
    result = {"main_image": main_img, "state": state}
    if collect_wrist and "robot0_eye_in_hand_image" in raw_obs:
        result["wrist_image"] = _rotate180(raw_obs["robot0_eye_in_hand_image"])
    return result


@dataclass
class LiberoTaskConfig:
    """Configuration for LIBERO environment."""
    task_suite_name: str = "libero_10"
    image_size: tuple[int, int] = (224, 224)
    action_dim: int = 7
    state_dim: int = 7
    max_episode_steps: int = 300
    has_wrist_camera: bool = True
    seed: int = 0

    # OffScreenRenderEnv params
    render_camera: str = "agentview"
    camera_heights: int = 128
    camera_widths: int = 128
    use_camera_obs: bool = True
    camera_depths: bool = False
    reward_shaping: bool = True

    # Reward
    reward_coef: float = 1.0
    use_rel_reward: bool = False


class LiberoRRFEnv(EmbodiedEnv):
    """LIBERO environment implementing EmbodiedEnv.

    Directly uses the upstream libero package (no RLinf dependency).

    Usage:
        cfg = {
            "task_suite_name": "libero_10",
            "image_size": (224, 224),
            "max_episode_steps": 300,
        }
        env = LiberoRRFEnv(cfg, num_envs=8, device="cuda:0")
        obs, info = env.reset()
        obs, rewards, dones, info = env.step(actions)
    """

    def __init__(self, cfg: dict, num_envs: int = 1, device: str = "cpu", **kwargs):
        defaults = LiberoTaskConfig()
        for k, v in defaults.__dict__.items():
            cfg.setdefault(k, v)

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.num_actions = cfg.get("action_dim", 7)
        self.num_obs = cfg.get("state_dim", 7)
        self.image_size = cfg.get("image_size", (224, 224))
        self.state_dim = self.num_obs
        self.has_wrist_camera = cfg.get("has_wrist_camera", True)
        self.max_episode_length = cfg.get("max_episode_steps", 300)
        self.num_privileged_obs = 0
        self._reward_coef = cfg.get("reward_coef", 1.0)

        # Buffers
        self.rew_buf = torch.zeros(num_envs, device=self.device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.obs_buf = None
        self.privileged_obs_buf = None
        self.extras = {}

        self._envs = []
        self._task_descriptions = []
        self._init_sim(cfg)

    def _init_sim(self, cfg: dict):
        """Initialize LIBERO environments via subprocess vector env."""
        try:
            from libero.libero import get_libero_path
            from libero.libero.benchmark import get_benchmark
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError:
            raise ImportError(
                "LIBERO not found. Install it:\n"
                "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO\n"
                "  pip install -e LIBERO\n"
                "  pip install robosuite"
            )

        bddl_root = get_libero_path("bddl_files")

        suite_name = cfg["task_suite_name"]
        benchmark = get_benchmark(suite_name)()
        num_tasks = benchmark.get_num_tasks()
        rng = np.random.default_rng(cfg.get("seed", 0))

        env_args = {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "render_camera": cfg.get("render_camera", "agentview"),
            "ignore_done": True,
            "use_camera_obs": cfg.get("use_camera_obs", True),
            "camera_depths": cfg.get("camera_depths", False),
            "camera_heights": cfg.get("camera_heights", 128),
            "camera_widths": cfg.get("camera_widths", 128),
            "reward_shaping": cfg.get("reward_shaping", True),
        }

        for i in range(self.num_envs):
            task_id = i % num_tasks
            task = benchmark.get_task(task_id)
            bddl_file = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl_file, **env_args)
            env.seed(int(rng.integers(0, 2**31)))
            self._envs.append(env)
            self._task_descriptions.append(task.language)

    def _batch_obs(self, raw_obs_list: list[dict]) -> dict[str, Any]:
        """Stack per-env observations into batched tensors."""
        extracted = [_extract_obs(o, self.has_wrist_camera) for o in raw_obs_list]
        result = {
            "main_images": torch.from_numpy(
                np.stack([e["main_image"] for e in extracted])
            ).to(self.device),
            "states": torch.from_numpy(
                np.stack([e["state"] for e in extracted])
            ).to(self.device),
            "task_descriptions": list(self._task_descriptions),
        }
        if self.has_wrist_camera and "wrist_image" in extracted[0]:
            result["wrist_images"] = torch.from_numpy(
                np.stack([e["wrist_image"] for e in extracted])
            ).to(self.device)
        return result

    # ---- EmbodiedEnv interface ----

    def get_observations(self) -> tuple[dict[str, Any], dict]:
        raw_obs_list = [env._get_observations() for env in self._envs]
        return self._batch_obs(raw_obs_list), self.extras

    def reset(self) -> tuple[dict[str, Any], dict]:
        raw_obs_list = []
        for env in self._envs:
            obs = env.reset()
            raw_obs_list.append(obs)
        obs = self._batch_obs(raw_obs_list)
        self.episode_length_buf.zero_()
        self.reset_buf.zero_()
        self.extras = {}
        return obs, self.extras

    def step(self, actions: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy()
        raw_obs_list, rewards_list, terms_list = [], [], []

        for i, env in enumerate(self._envs):
            obs, reward, done, info = env.step(actions_np[i])
            raw_obs_list.append(obs)
            rewards_list.append(reward)
            terms_list.append(done)

        obs = self._batch_obs(raw_obs_list)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        rewards = rewards * self._reward_coef
        terminated = torch.tensor(terms_list, dtype=torch.bool, device=self.device)
        self.episode_length_buf += 1
        truncated = self.episode_length_buf >= self.max_episode_length
        dones = terminated | truncated

        # Auto-reset done envs
        for i in range(self.num_envs):
            if dones[i]:
                reset_obs = self._envs[i].reset()
                raw_obs_list[i] = reset_obs
                self.episode_length_buf[i] = 0

        self.rew_buf = rewards
        self.reset_buf = dones
        self.extras = {"termination": terminated, "timeout": truncated}
        return obs, rewards, dones, self.extras

    def close(self):
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()
        self._envs = []
