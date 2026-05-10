"""Unitree G1 velocity-tracking AMP env configurations (MJLab).

Reuses :mod:`mjlab.tasks.velocity.config.g1` factories and attaches an ``amp``
observation group whose terms feed the AMP discriminator.

Reference: beyondAMP/source/amp_tasks_mjlab/amp_tasks_mjlab/velocity/g1/amp_env_cfg.py
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import (
    unitree_g1_flat_env_cfg,
    unitree_g1_rough_env_cfg,
)

# G1 anchor / key bodies used by the AMP discriminator.
G1_ANCHOR_NAME: str = "pelvis"
G1_KEY_BODY_NAMES: list[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
]

# AMP observation terms (joint-space). Matches ``AMPObsBaiscTerms`` in beyondAMP.
G1_AMP_OBS_TERMS: list[str] = ["joint_pos", "joint_vel"]


def _amp_obs_basic_terms() -> dict[str, ObservationTermCfg]:
    return {
        "joint_pos": ObservationTermCfg(func=joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel),
    }


def _amp_obs_basic_group() -> ObservationGroupCfg:
    return ObservationGroupCfg(
        terms=_amp_obs_basic_terms(),
        concatenate_terms=True,
        enable_corruption=False,
    )


def _attach_amp_group(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    cfg.observations["amp"] = _amp_obs_basic_group()
    return cfg


def unitree_g1_flat_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _attach_amp_group(unitree_g1_flat_env_cfg(play=play))


def unitree_g1_rough_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _attach_amp_group(unitree_g1_rough_env_cfg(play=play))
