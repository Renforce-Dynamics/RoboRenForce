"""
Visualize joint-space FK replay from a stored dynamics trajectory (pickle).

This script is analogous to `.references/beyondAMP/scripts/visualization/visualize_fk.py`,
but instead of using a custom MotionLoader and full root-state trajectories, it:

- Loads a pickled trajectory of joint positions (and optionally velocities).
- Replays these joint configurations on a chosen robot in Isaac Lab.
- Keeps the root pose fixed (no root xyz motion), i.e. only performs FK on joints.

Usage (example):

    python -m scripts.renforce.visualize_fk_from_pkl \\
        --pkl_file path/to/dynamics.pkl

The pickle file is expected to contain at least one of:

    - ``joint_pos``: Tensor/ndarray of shape [T, num_envs, num_joints] or [T, num_joints]
    - (optional) ``joint_vel``: same leading dimensions as ``joint_pos``

If ``num_envs`` is omitted, it is assumed to be 1.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch

from isaaclab.app import AppLauncher


# -------------------------------------------------------------------------- #
# CLI
# -------------------------------------------------------------------------- #

parser = argparse.ArgumentParser(description="Replay FK motions from a pickled joint trajectory.")
parser.add_argument("--pkl_file", type=str, required=True, help="Path to the dynamics pickle file.")

# Append AppLauncher CLI args (e.g., --experience, --renderer, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -------------------------------------------------------------------------- #
# Rest everything follows.
# -------------------------------------------------------------------------- #

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, G1_MINIMAL_CFG


@configclass
class FKReplaySceneCfg(InteractiveSceneCfg):
    """Configuration for an FK replay scene with a single robot."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                "kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )

    # robot: ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    


def _load_trajectory_dict(pkl_path: str) -> Dict[str, Any]:
    """Load a trajectory dict from a pickle file."""
    path = Path(pkl_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pickle file '{pkl_path}' does not exist.")
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Pickle file '{pkl_path}' must contain a dict, got {type(data)}.")
    return data


def _build_joint_trajectory_from_data(
    data: Dict[str, Any], num_joints: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build joint position/velocity trajectories from a generic dynamics dict.

    Priority:
        1. If 'joint_pos' (and optional 'joint_vel') exist, use them directly.
        2. Else, if 'pred_dynamic' exists, parse it using the layout:
           [root_pos(3), root_quat(4), root_lin_vel(3), root_ang_vel(3),
            joint_pos(num_joints), joint_vel(num_joints)].
    """
    # Case 1: direct joint_pos (+ optional joint_vel)
    if "joint_pos" in data:
        joint_pos = torch.as_tensor(data["joint_pos"], device=device, dtype=torch.float32)
        if joint_pos.ndim == 2:
            joint_pos = joint_pos.unsqueeze(1)
        elif joint_pos.ndim != 3:
            raise ValueError(
                f"'joint_pos' must have shape [T, num_joints] or [T, num_envs, num_joints], "
                f"but got shape {tuple(joint_pos.shape)}."
            )
        T, num_envs, nj = joint_pos.shape
        if nj != num_joints:
            raise ValueError(
                f"joint_pos num_joints={nj} does not match robot.num_joints={num_joints}."
            )
        if "joint_vel" in data:
            joint_vel = torch.as_tensor(data["joint_vel"], device=device, dtype=torch.float32)
            if joint_vel.ndim == 2:
                joint_vel = joint_vel.unsqueeze(1)
            if joint_vel.shape != joint_pos.shape:
                raise ValueError(
                    f"'joint_vel' shape {tuple(joint_vel.shape)} does not match 'joint_pos' "
                    f"shape {tuple(joint_pos.shape)}."
                )
        else:
            joint_vel = torch.zeros_like(joint_pos)
        return joint_pos, joint_vel

    # Case 2: parse from predicted dynamics (e.g. from MBPO save_example_from_replay)
    if "pred_dynamic" in data:
        dyn = torch.as_tensor(data["pred_dynamic"], device=device, dtype=torch.float32)
        # dyn: [B, T, D] or [T, D]
        if dyn.ndim == 2:
            dyn = dyn.unsqueeze(0)
        if dyn.ndim != 3:
            raise ValueError(
                f"'pred_dynamic' must have shape [T, D] or [B, T, D], got {tuple(dyn.shape)}."
            )
        B, T, D = dyn.shape
        expected_dim = 9 + 2 * num_joints
        if D != expected_dim:
            raise ValueError(
                f"pred_dynamic dim={D} does not match expected layout 9 + 2 * num_joints={expected_dim}."
            )
        dyn0 = dyn[0]  # [T, D]
        joint_pos = dyn0[:, 9 : 9 + num_joints]
        joint_vel = dyn0[:, 9 + num_joints : 9 + 2 * num_joints]
        joint_pos = joint_pos.unsqueeze(1)  # [T, 1, num_joints]
        joint_vel = joint_vel.unsqueeze(1)  # [T, 1, num_joints]
        return joint_pos, joint_vel

    raise KeyError("Unsupported trajectory format: expected 'joint_pos' or 'pred_dynamic' in pickle data.")


def run_simulator(sim: SimulationContext, scene: InteractiveScene, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
    """Main FK replay loop."""
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    T, num_envs_traj, num_joints = joint_pos.shape
    if scene.num_envs != num_envs_traj:
        raise ValueError(
            f"Scene num_envs={scene.num_envs} does not match trajectory num_envs={num_envs_traj}."
        )
    if robot.num_joints != num_joints:
        raise ValueError(
            f"Robot num_joints={robot.num_joints} does not match trajectory num_joints={num_joints}."
        )

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    import tqdm

    pbar = tqdm.tqdm()

    default_root_states = robot.data.default_root_state.clone()
    # Use the robot's default joint positions as initialization (matches Unitree A1 cfg).
    default_joint_pos = robot.data.default_joint_pos.clone()

    while simulation_app.is_running():
        time_steps += 1
        pbar.update(1)

        reset_ids = time_steps >= T
        time_steps[reset_ids] = 0

        root_states = default_root_states.clone()

        cur_joint_delta = joint_pos[time_steps, torch.arange(scene.num_envs, device=sim.device)]
        cur_joint_pos = default_joint_pos + cur_joint_delta
        cur_joint_vel = joint_vel[time_steps, torch.arange(scene.num_envs, device=sim.device)]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(cur_joint_pos, cur_joint_vel)

        scene.write_data_to_sim()

        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(
            pos_lookat + np.array([2.0, 2.0, 0.5]),
            pos_lookat,
        )


def main() -> None:
    device = torch.device(args_cli.device)

    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt = 1 / 20
    )
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    # First create a temporary scene to know robot.num_joints.
    scene_cfg = FKReplaySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    data = _load_trajectory_dict(args_cli.pkl_file)
    joint_pos, joint_vel = _build_joint_trajectory_from_data(
        data, num_joints=robot.num_joints, device=device
    )

    run_simulator(sim, scene, joint_pos, joint_vel)


if __name__ == "__main__":
    main()
    simulation_app.close()

