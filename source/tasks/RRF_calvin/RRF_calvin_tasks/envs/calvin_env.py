"""
CALVIN Environment Wrapper for RoboRenForce

Directly wraps the CALVIN benchmark behind the EmbodiedEnv interface.
No dependency on RLinf — uses upstream calvin_env package directly.

CALVIN evaluates long-horizon manipulation via 5-subtask sequences.
Each subtask is a natural language instruction.

Task suites:
    - calvin_d:    scene D only
    - calvin_abc:  scenes A, B, C
    - calvin_abcd: all four scenes

Observation format:
    main_images:      [B, H, W, 3] - static camera RGB
    wrist_images:     [B, H, W, 3] - gripper camera RGB
    states:           [B, 7]       - 7-DOF joint positions
    task_descriptions: list[str]   - current subtask instruction

Action format: [B, 7] — 6D EEF pose + 1D gripper

Dependencies:
    Follow CALVIN setup: https://github.com/mees/calvin
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from RoboRenForce.prototype.embodied import EmbodiedEnv


@dataclass
class CalvinTaskConfig:
    """Configuration for CALVIN environment."""
    task_suite_name: str = "calvin_abcd"
    image_size: tuple[int, int] = (224, 224)
    action_dim: int = 7
    state_dim: int = 7
    max_episode_steps: int = 360
    has_wrist_camera: bool = True
    num_subtasks: int = 5
    seed: int = 0
    dataset_path: str = ""
    reward_coef: float = 1.0


class CalvinRRFEnv(EmbodiedEnv):
    """CALVIN environment implementing EmbodiedEnv.

    Directly uses upstream calvin_env package (no RLinf dependency).

    Usage:
        cfg = {
            "task_suite_name": "calvin_abcd",
            "dataset_path": "/path/to/calvin/dataset",
            "max_episode_steps": 360,
        }
        env = CalvinRRFEnv(cfg, num_envs=4, device="cuda:0")
        obs, info = env.reset()
        obs, rewards, dones, info = env.step(actions)
    """

    def __init__(self, cfg: dict, num_envs: int = 1, device: str = "cpu", **kwargs):
        defaults = CalvinTaskConfig()
        for k, v in defaults.__dict__.items():
            cfg.setdefault(k, v)

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.num_actions = cfg.get("action_dim", 7)
        self.num_obs = cfg.get("state_dim", 7)
        self.image_size = cfg.get("image_size", (224, 224))
        self.state_dim = self.num_obs
        self.has_wrist_camera = True
        self.max_episode_length = cfg.get("max_episode_steps", 360)
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
        self._task_descriptions = ["manipulation task"] * num_envs
        self._init_sim(cfg)

    def _init_sim(self, cfg: dict):
        """Initialize CALVIN environments."""
        try:
            import hydra
            from calvin_env.envs.play_table_env import PlayTableSimEnv
        except ImportError:
            raise ImportError(
                "CALVIN not found. Install it:\n"
                "  git clone --recurse-submodules https://github.com/mees/calvin\n"
                "  pip install -e calvin/calvin_env"
            )

        import os
        dataset_path = cfg.get("dataset_path", "")
        env_config_path = os.path.join(dataset_path, ".hydra", "config.yaml") if dataset_path else ""

        if dataset_path and os.path.exists(env_config_path):
            with hydra.initialize_config_dir(config_dir=os.path.dirname(env_config_path), version_base=None):
                env_cfg = hydra.compose(config_name="config")
        else:
            # Fall back to upstream calvin_env data-collection config so the env
            # is constructible without a recorded dataset (useful for smoke /
            # CI). For real RL training, point dataset_path at a CALVIN release.
            import calvin_env
            calvin_conf_dir = os.path.join(
                os.path.dirname(calvin_env.__file__), "..", "conf"
            )
            calvin_conf_dir = os.path.abspath(calvin_conf_dir)
            with hydra.initialize_config_dir(config_dir=calvin_conf_dir, version_base=None):
                env_cfg = hydra.compose(
                    config_name="config_data_collection",
                    overrides=["cameras=static_and_gripper"],
                )
            env_cfg.env["use_egl"] = False
            env_cfg.env["show_gui"] = False
            env_cfg.env["use_vr"] = False
            env_cfg.env["use_scene_info"] = True

        # PlayTableSimEnv expects DictConfig nodes (attribute access like
        # ``cameras[name].width``). Resolve interpolations against the full
        # config tree first so that ``cameras: ${cameras}`` etc. become
        # concrete sub-nodes; then pull env keys out as DictConfig.
        from omegaconf import OmegaConf
        OmegaConf.resolve(env_cfg)
        env_node = env_cfg.env
        OmegaConf.set_struct(env_node, False)
        for marker in ("_target_", "_recursive_"):
            if marker in env_node:
                del env_node[marker]
        env_kwargs = {k: env_node[k] for k in env_node}
        for i in range(self.num_envs):
            env = PlayTableSimEnv(**env_kwargs)
            self._envs.append(env)

    def _extract_obs(self, raw_obs: dict) -> dict:
        """Extract standard obs from single CALVIN env observation."""
        return {
            "main_image": raw_obs["rgb_obs"]["rgb_static"],
            "wrist_image": raw_obs["rgb_obs"]["rgb_gripper"],
            "state": raw_obs["robot_obs"][:7].astype(np.float32),
        }

    def _batch_obs(self, raw_obs_list: list[dict]) -> dict[str, Any]:
        """Stack per-env observations into batched tensors."""
        extracted = [self._extract_obs(o) for o in raw_obs_list]
        result = {
            "main_images": torch.from_numpy(
                np.stack([e["main_image"] for e in extracted])
            ).to(self.device),
            "wrist_images": torch.from_numpy(
                np.stack([e["wrist_image"] for e in extracted])
            ).to(self.device),
            "states": torch.from_numpy(
                np.stack([e["state"] for e in extracted])
            ).to(self.device),
            "task_descriptions": list(self._task_descriptions),
        }
        return result

    # ---- EmbodiedEnv interface ----

    def get_observations(self) -> tuple[dict[str, Any], dict]:
        raw_obs_list = [env.get_obs() for env in self._envs]
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
            # CALVIN expects gripper to be discrete (-1 close / 1 open). The
            # policy outputs a continuous scalar — threshold at 0 to convert.
            a = actions_np[i].copy()
            if a.shape[-1] >= 7:
                a[..., 6] = 1.0 if a[..., 6] >= 0 else -1.0
            obs, reward, done, info = env.step(a)
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
