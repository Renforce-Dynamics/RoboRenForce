"""
RoboTwin Environment Wrapper

Wraps the RoboTwin simulator (SAPIEN 3-based) behind the EmbodiedEnv interface.

RoboTwin provides bimanual robotic manipulation tasks with 50+ scenarios.
The upstream env is `robotwin.envs.vector_env.VectorEnv`, which manages
parallel SAPIEN simulation processes.

Reference: RLinf rlinf/envs/robotwin/robotwin_env.py

Dependencies:
    pip install sapien==3.0.1 mplib==0.2.1
    git clone https://github.com/RoboTwin-Platform/RoboTwin -b RLinf_support
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from RoboRenForce.prototype.embodied import EmbodiedEnv


def _center_crop(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Center-crop image to target (H, W)."""
    h, w = image.shape[:2]
    th, tw = target_size
    if h == th and w == tw:
        return image
    y0 = max(0, (h - th) // 2)
    x0 = max(0, (w - tw) // 2)
    return image[y0 : y0 + th, x0 : x0 + tw]


@dataclass
class RoboTwinTaskConfig:
    """Upstream RoboTwin task configuration (passed to VectorEnv)."""
    task_name: str = "place_empty_cup"
    planner_backend: str = "mplib"              # "mplib" or "curobo"
    embodiment: list = field(default_factory=lambda: ["piper", "piper", 0.6])
    step_lim: int = 200
    domain_randomization: bool = False
    assets_path: str = ""                       # path to RoboTwin assets
    seeds_path: str = ""                        # path to seeds JSON dir

    # Camera
    camera_width: int = 320
    camera_height: int = 240
    collect_wrist_camera: bool = False

    # Data types
    data_type_qpos: bool = True
    data_type_qvel: bool = False


class RoboTwinEnv(EmbodiedEnv):
    """RoboTwin environment implementing the EmbodiedEnv interface.

    Wraps `robotwin.envs.vector_env.VectorEnv` which manages N parallel
    SAPIEN simulation processes for bimanual manipulation tasks.

    Usage:
        cfg = {
            "task_config": RoboTwinTaskConfig(task_name="place_empty_cup"),
            "image_size": (224, 224),
            "action_dim": 14,
            "state_dim": 14,
            "max_episode_steps": 200,
            "use_custom_reward": True,
            "reward_coef": 5.0,
        }
        env = RoboTwinEnv(cfg, num_envs=8, device="cuda:0")
        obs, info = env.reset()
        obs, rewards, dones, info = env.step(actions)
    """

    def __init__(self, cfg: dict, num_envs: int = 1, device: str = "cpu", **kwargs):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        # Parse config
        task_cfg = cfg.get("task_config", RoboTwinTaskConfig())
        if isinstance(task_cfg, dict):
            task_cfg = RoboTwinTaskConfig(**task_cfg)
        self.task_cfg = task_cfg

        self.num_actions = cfg.get("action_dim", 14)  # bimanual: 7 per arm
        self.num_obs = cfg.get("state_dim", 14)
        self.image_size = cfg.get("image_size", (224, 224))
        self.state_dim = self.num_obs
        self.has_wrist_camera = task_cfg.collect_wrist_camera
        self.max_episode_length = cfg.get("max_episode_steps", task_cfg.step_lim)
        self.num_privileged_obs = 0

        # Reward config
        self._use_custom_reward = cfg.get("use_custom_reward", True)
        self._use_rel_reward = cfg.get("use_rel_reward", True)
        self._reward_coef = cfg.get("reward_coef", 5.0)
        self._center_crop = cfg.get("center_crop", True)

        # Task description
        self._task_description = cfg.get(
            "task_description",
            task_cfg.task_name.replace("_", " "),
        )

        # Seed management
        self._train_seeds = None
        self._eval_seeds = None
        if task_cfg.seeds_path:
            seeds_dir = Path(task_cfg.seeds_path)
            train_path = seeds_dir / "train_seeds.json"
            eval_path = seeds_dir / "eval_seeds.json"
            if train_path.exists():
                self._train_seeds = json.loads(train_path.read_text())
            if eval_path.exists():
                self._eval_seeds = json.loads(eval_path.read_text())

        # Tracking buffers
        self.rew_buf = torch.zeros(num_envs, device=self.device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._prev_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.obs_buf = None
        self.privileged_obs_buf = None
        self.extras = {}

        # Initialize upstream sim
        self._venv = None
        self._init_sim()

    def _init_sim(self):
        """Initialize the RoboTwin VectorEnv."""
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass  # already set

        try:
            from robotwin.envs.vector_env import VectorEnv
        except ImportError:
            raise ImportError(
                "RoboTwin simulator not found. Install it:\n"
                "  git clone https://github.com/RoboTwin-Platform/RoboTwin -b RLinf_support\n"
                "  export PYTHONPATH=/path/to/RoboTwin:$PYTHONPATH\n"
                "  pip install sapien==3.0.1 mplib==0.2.1"
            )

        # RoboTwin upstream `_base_task.setup_scene` unconditionally selects the
        # ray-tracing shader pipeline (`set_camera_shader_dir("rt")`) and the
        # OIDN denoiser. On hosts without a working RT-capable Vulkan ICD
        # (e.g. when forced onto Mesa LLVMpipe via `VK_ICD_FILENAMES=lvp_icd.json`)
        # this hangs forever in the Vulkan command buffer. Force the default
        # rasterization shader unless the user explicitly opts back into RT.
        if os.environ.get("RRF_ROBOTWIN_DISABLE_RT", "1") != "0":
            try:
                import sapien.render as _sr
                _sr.set_camera_shader_dir = lambda *_, **__: None
                _sr.set_ray_tracing_samples_per_pixel = lambda *_, **__: None
                _sr.set_ray_tracing_path_depth = lambda *_, **__: None
                _sr.set_ray_tracing_denoiser = lambda *_, **__: None
            except Exception:
                pass

        task_config = {
            "task_name": self.task_cfg.task_name,
            "planner_backend": self.task_cfg.planner_backend,
            "embodiment": self.task_cfg.embodiment,
            "step_lim": self.task_cfg.step_lim,
            "domain_randomization": self.task_cfg.domain_randomization,
            "save_path": str(Path(self.cfg.get("save_path", "/tmp/robotwin_logs"))),
        }
        if self.task_cfg.assets_path:
            task_config["assets_path"] = self.task_cfg.assets_path

        task_config["camera"] = {
            "width": self.task_cfg.camera_width,
            "height": self.task_cfg.camera_height,
            "collect_wrist_camera": self.task_cfg.collect_wrist_camera,
        }
        task_config["data_type"] = {
            "qpos": self.task_cfg.data_type_qpos,
            "qvel": self.task_cfg.data_type_qvel,
        }

        seeds = list(range(self.num_envs))
        if self._train_seeds is not None:
            seeds = self._train_seeds[: self.num_envs]

        self._venv = VectorEnv(
            task_config=task_config,
            n_envs=self.num_envs,
            env_seeds=seeds,
        )

    def _fit_state_dim(self, states: torch.Tensor) -> torch.Tensor:
        """Clip/pad the last dim of `states` to `self.state_dim`.

        RoboTwin upstream qpos width depends on the chosen embodiment (single
        arm vs bimanual + gripper), so policies that hard-code state_dim
        crash on mismatch unless we normalize here.
        """
        if states.shape[-1] > self.state_dim:
            return states[..., : self.state_dim]
        if states.shape[-1] < self.state_dim:
            pad = torch.zeros(
                *states.shape[:-1], self.state_dim - states.shape[-1],
                dtype=states.dtype, device=states.device,
            )
            return torch.cat([states, pad], dim=-1)
        return states

    def _extract_obs(self, raw_obs_list: list[dict]) -> dict[str, Any]:
        """Convert per-env observations to batched obs dict."""
        main_images = []
        wrist_images = []
        states = []

        for obs in raw_obs_list:
            img = obs.get("full_image", obs.get("image"))
            if isinstance(img, np.ndarray) and self._center_crop:
                img = _center_crop(img, self.image_size)
            main_images.append(img)

            if self.has_wrist_camera:
                left_wrist = obs.get("left_wrist_image")
                right_wrist = obs.get("right_wrist_image")
                if left_wrist is not None and right_wrist is not None:
                    wrist_img = np.concatenate([left_wrist, right_wrist], axis=1)
                    if self._center_crop:
                        wrist_img = _center_crop(wrist_img, self.image_size)
                    wrist_images.append(wrist_img)

            state = obs.get("state", obs.get("qpos"))
            if state is not None:
                if isinstance(state, np.ndarray):
                    states.append(state)
                else:
                    states.append(np.array(state, dtype=np.float32))

        result = {
            "main_images": torch.from_numpy(np.stack(main_images)).to(self.device),
            "task_descriptions": [self._task_description] * self.num_envs,
        }
        if states:
            states_t = torch.from_numpy(np.stack(states)).float().to(self.device)
            result["states"] = self._fit_state_dim(states_t)
        if wrist_images:
            result["wrist_images"] = torch.from_numpy(np.stack(wrist_images)).to(self.device)

        return result

    def _calc_reward(self, terminations: torch.Tensor) -> torch.Tensor:
        """Compute shaped reward from binary success signal."""
        if not self._use_custom_reward:
            return terminations.float()

        if self._use_rel_reward:
            new_success = terminations & ~self._prev_success
            reward = self._reward_coef * new_success.float()
            self._prev_success = terminations.clone()
            return reward

        return self._reward_coef * terminations.float()

    # ---- EmbodiedEnv interface ----

    def get_observations(self) -> tuple[dict[str, Any], dict]:
        if self._venv is None:
            raise RuntimeError("Simulator not initialized. Call reset() first.")
        raw_obs = self._venv.get_obs()
        obs = self._extract_obs(raw_obs)
        return obs, self.extras

    def reset(self) -> tuple[dict[str, Any], dict]:
        if self._venv is None:
            self._init_sim()

        raw_obs_list, infos = self._venv.reset()
        if not isinstance(raw_obs_list, list):
            raw_obs_list = [raw_obs_list]

        obs = self._extract_obs(raw_obs_list)
        self.episode_length_buf.zero_()
        self._prev_success.zero_()
        self.extras = {"infos": infos}
        return obs, self.extras

    def step(self, actions: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy()
        raw_obs_list, raw_rewards, terminations_np, truncations_np, infos = self._venv.step(actions_np)

        if not isinstance(raw_obs_list, list):
            raw_obs_list = [raw_obs_list]

        obs = self._extract_obs(raw_obs_list)
        terminations = torch.from_numpy(np.array(terminations_np, dtype=np.bool_)).to(self.device)
        truncations = torch.from_numpy(np.array(truncations_np, dtype=np.bool_)).to(self.device)
        dones = terminations | truncations
        rewards = self._calc_reward(terminations)

        self.episode_length_buf += 1
        self.rew_buf = rewards
        self.reset_buf = dones
        self.extras = {
            "termination": terminations,
            "timeout": truncations,
            "infos": infos,
        }
        return obs, rewards, dones, self.extras

    def chunk_step(
        self,
        actions: torch.Tensor,
    ) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """Step with action chunks [B, chunk_len, action_dim].

        For action-chunked VLA policies (diffusion, pi0, etc.).
        """
        B, chunk_len, action_dim = actions.shape
        actions_np = actions.detach().cpu().numpy()

        raw_obs_list, raw_rewards, term_list, trunc_list, infos_list = self._venv.chunk_step(actions_np)

        obs_list = []
        for t in range(chunk_len):
            step_obs = [raw_obs_list[t][i] if isinstance(raw_obs_list[t], list)
                        else raw_obs_list[t] for i in range(B)]
            obs_list.append(self._extract_obs(step_obs))

        chunk_rewards = torch.tensor(raw_rewards, dtype=torch.float32, device=self.device)
        chunk_terminations = torch.tensor(term_list, dtype=torch.bool, device=self.device)
        chunk_truncations = torch.tensor(trunc_list, dtype=torch.bool, device=self.device)

        return obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list

    def close(self):
        if self._venv is not None:
            if hasattr(self._venv, "close"):
                self._venv.close()
            self._venv = None
