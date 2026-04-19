"""
Tests for RRF_robotwin_tasks package.

Tests that don't require the simulator (import checks, config, obs extraction).
"""

import sys
import pytest
import numpy as np
import torch


# ---- Import tests ----

def test_import_robotwin_env():
    from RRF_robotwin_tasks.envs.robotwin_env import RoboTwinEnv, RoboTwinTaskConfig
    assert RoboTwinEnv is not None
    assert RoboTwinTaskConfig is not None


def test_import_robots():
    from RRF_robotwin_tasks.robots import Piper, PiperCfg, AlohaAgilex, AlohaAgilexCfg
    assert PiperCfg().action_dim == 14
    assert AlohaAgilexCfg().action_dim == 14


def test_import_datasets():
    from RRF_robotwin_tasks.datasets import RoboTwinDemoCfg
    cfg = RoboTwinDemoCfg(task_name="pick_apple")
    assert cfg.source == "robotwin"
    assert "pick_apple" in cfg.tags


# ---- Config tests ----

def test_task_config_defaults():
    from RRF_robotwin_tasks.envs.robotwin_env import RoboTwinTaskConfig
    cfg = RoboTwinTaskConfig()
    assert cfg.task_name == "place_empty_cup"
    assert cfg.planner_backend == "mplib"
    assert cfg.embodiment == ["piper", "piper", 0.6]
    assert cfg.step_lim == 200


def test_task_config_custom():
    from RRF_robotwin_tasks.envs.robotwin_env import RoboTwinTaskConfig
    cfg = RoboTwinTaskConfig(
        task_name="click_bell",
        planner_backend="curobo",
        collect_wrist_camera=True,
    )
    assert cfg.task_name == "click_bell"
    assert cfg.collect_wrist_camera is True


# ---- Robot tests ----

def test_piper_repack():
    from RRF_robotwin_tasks.robots.piper import Piper, PiperCfg
    robot = Piper(PiperCfg())
    action = torch.randn(14)
    repacked = robot.repack_action(action)
    assert torch.equal(action, repacked)  # pass-through for piper


def test_piper_joint_names():
    from RRF_robotwin_tasks.robots.piper import PIPER_JOINT_NAMES
    assert len(PIPER_JOINT_NAMES) == 14
    assert "left_joint1" in PIPER_JOINT_NAMES
    assert "right_gripper" in PIPER_JOINT_NAMES


# ---- Obs extraction test (no sim needed) ----

def test_center_crop():
    from RRF_robotwin_tasks.envs.robotwin_env import _center_crop
    img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    cropped = _center_crop(img, (224, 224))
    assert cropped.shape == (224, 224, 3)


def test_center_crop_noop():
    from RRF_robotwin_tasks.envs.robotwin_env import _center_crop
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cropped = _center_crop(img, (224, 224))
    assert cropped is img  # same object, no copy needed


def test_env_init_fails_without_sim():
    """RoboTwinEnv should raise ImportError when robotwin is not installed."""
    from RRF_robotwin_tasks.envs.robotwin_env import RoboTwinEnv
    cfg = {"action_dim": 14, "state_dim": 14}
    with pytest.raises(ImportError, match="RoboTwin simulator not found"):
        env = RoboTwinEnv(cfg, num_envs=1)


# ---- Setup.py sanity ----

def test_setup_py_importable():
    """Verify the package structure is correct."""
    import RRF_robotwin_tasks
    assert hasattr(RRF_robotwin_tasks, "__name__")
