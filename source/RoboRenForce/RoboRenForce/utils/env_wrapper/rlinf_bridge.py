"""
RLinf-to-RoboRenForce Bridge Adapter

Wraps any RLinf environment (LiberoEnv, ManiskillEnv, CalvinEnv, D4RLEnv)
behind the RoboRenForce EmbodiedEnv interface.

RLinf envs follow a common pattern:
    - reset()  -> (obs_dict, infos)       obs_dict has main_images, states, etc.
    - step()   -> (obs_dict, rew, term, trunc, infos)
    - chunk_step() -> (obs_list, rew, term, trunc, infos_list)

RoboRenForce EmbodiedEnv expects:
    - reset()  -> (obs_dict, extras)
    - step()   -> (obs_dict, rewards, dones, extras)

This bridge handles the 5-tuple → 4-tuple conversion and buffer management.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from RoboRenForce.prototype.embodied import EmbodiedEnv


class RLinfBridgeEnv(EmbodiedEnv):
    """Bridge adapter: wraps an RLinf env class behind EmbodiedEnv.

    Subclasses only need to:
        1. Call super().__init__() with the right params
        2. Set self._rlinf_env to the constructed RLinf env instance
        3. Optionally override _post_process_obs() for env-specific tweaks

    Example:
        class LiberoRRFEnv(RLinfBridgeEnv):
            def __init__(self, cfg, num_envs, device, **kw):
                super().__init__(cfg, num_envs, device, **kw)
                from rlinf.envs.libero.libero_env import LiberoEnv
                self._rlinf_env = LiberoEnv(rlinf_cfg, num_envs, ...)
    """

    def __init__(self, cfg: dict, num_envs: int = 1, device: str = "cpu", **kwargs):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        self.num_actions = cfg.get("action_dim", 7)
        self.num_obs = cfg.get("state_dim", 7)
        self.image_size = cfg.get("image_size", (224, 224))
        self.state_dim = self.num_obs
        self.has_wrist_camera = cfg.get("has_wrist_camera", False)
        self.max_episode_length = cfg.get("max_episode_steps", 300)
        self.num_privileged_obs = 0

        # Tracking buffers
        self.rew_buf = torch.zeros(num_envs, device=self.device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.obs_buf = None
        self.privileged_obs_buf = None
        self.extras = {}

        # To be set by subclass
        self._rlinf_env = None

    def _to_device(self, obs_dict: dict) -> dict:
        """Move obs tensors to target device."""
        result = {}
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, np.ndarray):
                result[k] = torch.from_numpy(v).to(self.device)
            else:
                result[k] = v  # list[str] for task_descriptions
        return result

    def _post_process_obs(self, obs_dict: dict) -> dict:
        """Override in subclass for env-specific obs transformations."""
        return obs_dict

    # ---- EmbodiedEnv interface ----

    def get_observations(self) -> tuple[dict[str, Any], dict]:
        obs, infos = self._rlinf_env.reset()
        obs = self._to_device(obs)
        obs = self._post_process_obs(obs)
        return obs, self.extras

    def reset(self) -> tuple[dict[str, Any], dict]:
        obs, infos = self._rlinf_env.reset()
        obs = self._to_device(obs)
        obs = self._post_process_obs(obs)

        self.episode_length_buf.zero_()
        self.reset_buf.zero_()
        self.extras = {"infos": infos}
        return obs, self.extras

    def step(self, actions: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict]:
        # RLinf envs accept numpy actions
        actions_np = actions.detach().cpu().numpy()

        # RLinf returns 5-tuple: (obs, reward, terminated, truncated, infos)
        obs, rewards, terminated, truncated, infos = self._rlinf_env.step(actions_np)

        obs = self._to_device(obs)
        obs = self._post_process_obs(obs)

        # Convert to tensors
        if isinstance(rewards, (np.ndarray, list)):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        elif isinstance(rewards, torch.Tensor):
            rewards = rewards.to(self.device)

        if isinstance(terminated, (np.ndarray, list)):
            terminated = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        if isinstance(truncated, (np.ndarray, list)):
            truncated = torch.tensor(truncated, dtype=torch.bool, device=self.device)

        dones = terminated | truncated

        self.episode_length_buf += 1
        self.rew_buf = rewards
        self.reset_buf = dones
        self.extras = {
            "termination": terminated,
            "timeout": truncated,
            "infos": infos,
        }
        return obs, rewards, dones, self.extras

    def chunk_step(
        self, actions: torch.Tensor
    ) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """Action-chunked stepping for VLA policies. [B, chunk_len, action_dim]."""
        if not hasattr(self._rlinf_env, "chunk_step"):
            # Fallback: step one at a time
            B, chunk_len, _ = actions.shape
            obs_list, infos_list = [], []
            all_rewards, all_terms, all_truncs = [], [], []
            for t in range(chunk_len):
                obs, rew, dones, extras = self.step(actions[:, t])
                obs_list.append(obs)
                all_rewards.append(rew)
                all_terms.append(extras.get("termination", dones))
                all_truncs.append(extras.get("timeout", torch.zeros_like(dones)))
                infos_list.append(extras.get("infos", {}))
            return (
                obs_list,
                torch.stack(all_rewards, dim=1),
                torch.stack(all_terms, dim=1),
                torch.stack(all_truncs, dim=1),
                infos_list,
            )

        actions_np = actions.detach().cpu().numpy()
        obs_list, rewards, terms, truncs, infos_list = self._rlinf_env.chunk_step(actions_np)

        processed_obs = [self._post_process_obs(self._to_device(o)) for o in obs_list]

        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).float().to(self.device)
        if isinstance(terms, np.ndarray):
            terms = torch.from_numpy(terms).bool().to(self.device)
        if isinstance(truncs, np.ndarray):
            truncs = torch.from_numpy(truncs).bool().to(self.device)

        return processed_obs, rewards, terms, truncs, infos_list

    def close(self):
        if self._rlinf_env is not None and hasattr(self._rlinf_env, "close"):
            self._rlinf_env.close()
