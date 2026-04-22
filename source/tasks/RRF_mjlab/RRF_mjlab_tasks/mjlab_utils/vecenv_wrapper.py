"""RoboRenForce wrapper for MJLab's ManagerBasedRlEnv.

Maps MJLab's observation group convention ("actor"/"critic") to RoboRenForce's
("policy"/"critic"), and adapts the 5-tuple step return to the 4-tuple
(obs, rew, dones, extras) expected by RoboRenForce runners.
"""

import gymnasium as gym
import torch

from RoboRenForce.utils.env_wrapper import RoboRenForceVecEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# MJLab uses "actor" where IsaacLab uses "policy"
_MJLAB_POLICY_KEY = "actor"


class _ObsSpaceDict(dict):
    """Thin dict wrapper that provides .shape access on values for dim_params.

    RRF's RFDynamicEnvWrapper.dim_params accesses observation_space[key].shape,
    so values must expose a .shape attribute. MJLab's Box already does.
    """

    def get(self, key, default=None):
        return super().get(key, default)


class RoboRenForceMJLabEnvWrapper(RoboRenForceVecEnv):
    """Wraps MJLab ManagerBasedRlEnv for RoboRenForce runners.

    Mirrors RoboRenForceLabEnvWrapper but targets MJLab's API:
    - Observation groups: "actor" (→ policy), "critic"
    - Step returns: (obs_dict, reward, terminated, truncated, extras)
    - Managers: observation_manager, action_manager, command_manager, reward_manager
    """

    def __init__(self, env: "ManagerBasedRlEnv", clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions

        # Store core env info
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # Dimensions
        self.num_actions = self.unwrapped.action_manager.total_action_dim

        self.num_obs = self.unwrapped.observation_manager.group_obs_dim[_MJLAB_POLICY_KEY][0]

        if "critic" in self.unwrapped.observation_manager.group_obs_dim:
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        else:
            self.num_privileged_obs = 0

        # Modify action space if clipping
        self._modify_action_space()

        # Reset at init (runner does not call reset)
        self.env.reset()

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    # -- Properties --

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self):
        """Returns observation space dict with 'actor' remapped to 'policy'.

        Returns a plain dict-like object since mjlab's Box type is not
        compatible with gymnasium's Dict space.
        """
        space = self.env.observation_space
        if hasattr(space, 'spaces'):
            raw = space.spaces
        elif hasattr(space, 'items'):
            raw = dict(space)
        else:
            return space
        remapped = {}
        for k, v in raw.items():
            remapped["policy" if k == _MJLAB_POLICY_KEY else k] = v
        return _ObsSpaceDict(remapped)

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> "ManagerBasedRlEnv":
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    # -- MDP operations --

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def _remap_obs(self, obs_dict: dict) -> dict:
        """Remap MJLab 'actor' key to RRF 'policy' key."""
        remapped = {}
        for k, v in obs_dict.items():
            remapped["policy" if k == _MJLAB_POLICY_KEY else k] = v
        return remapped

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs_dict = self.unwrapped.observation_manager.compute()
        obs_dict = self._remap_obs(obs_dict)
        return obs_dict["policy"], {"observations": obs_dict}

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, _ = self.env.reset()
        obs_dict = self._remap_obs(obs_dict)
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs_dict = self._remap_obs(obs_dict)

        dones = (terminated | truncated).to(dtype=torch.long)
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return obs, rew, dones, extras

    def close(self):
        return self.env.close()

    # -- Helpers --

    def _modify_action_space(self):
        if self.clip_actions is None:
            return
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
