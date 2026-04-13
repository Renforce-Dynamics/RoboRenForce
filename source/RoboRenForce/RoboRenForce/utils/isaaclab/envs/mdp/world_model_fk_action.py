from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Optional, Sequence, Dict, Any

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from RoboRenForce.utils.env_wrapper.lab_wrapper.world_model.imagine_env_wrapper import RFImagineEnvWrapper
    from RoboRenForce.components.world_models.system_dynamics.system_dynamics_mlp import SystemDynamicsMLP


class WorldModelFKAction(ActionTerm):
    """Action term that uses a learned system dynamics world model to drive FK state.

    High-level idea:
    - The raw actions correspond to the control commands from the RL policy (shape: (num_envs, action_dim)).
    - The action term maintains its own history of dynamic states and actions.
    - On (decimation) ticks, it rolls out the world model to predict the next dynamic state.
    - The predicted dynamic state is parsed into root pose and joint states and written into the
      underlying Isaac Lab environment's state buffers (NOT directly to the simulator).

    Notes
    -----
    - This class is intentionally generic and environment-agnostic.
    - The mapping from the flat dynamic vector to concrete robot states is handled in
      :meth:`_parse_dynamic`.
    - The mapping from parsed state to environment buffers is handled in
      :meth:`_apply_parsed_state_to_env`, which may need to be overridden for specific robots/tasks.
    """

    cfg: "WorldModelFKActionCfg"

    def __init__(self, cfg: "WorldModelFKActionCfg", env: "ManagerBasedRLEnv") -> None:
        # Initialize ActionTerm base.
        super().__init__(cfg, env)

        self.env: "ManagerBasedRLEnv" = env
        # Resolve robot articulation from the scene.
        self.robot: Articulation = env.scene[cfg.asset_name]

        self._env_wrapper: Optional["RFImagineEnvWrapper"] = None
        self.system_dynamics: Optional["SystemDynamicsMLP"] = None

        # Raw high-level actions coming from the policy.
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # World-model specific dimensions and history buffers.
        self._dynamic_dim: Optional[int] = None
        self._world_action_dim: Optional[int] = None
        self._history_horizon: Optional[int] = None
        self._dynamic_history: Optional[torch.Tensor] = None  # [num_envs, H, dynamic_dim]
        self._action_history: Optional[torch.Tensor] = None  # [num_envs, H, world_action_dim]

        # Decimation counter and last predicted state cache.
        self._counter: int = 0
        self._last_state_cache: Optional[Dict[str, torch.Tensor]] = None

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #

    @property
    def action_dim(self) -> int:
        """Dimension of the high-level action.

        This is typically the action dimension of the RL policy and should be
        configured in :class:`WorldModelFKActionCfg`.
        """
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # No extra processing at this level; subclasses can override if needed.
        return self._raw_actions

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def set_env_wrapper(self, env_wrapper: "RFImagineEnvWrapper") -> None:
        """Attach the RFImagineEnvWrapper and initialize world-model-related buffers.

        This method is intended to be called *after* the Isaac Lab environment is
        created and wrapped by :class:`RFImagineEnvWrapper`.
        """
        self._env_wrapper = env_wrapper

        # Get world model instance from wrapper.
        system_dynamics = env_wrapper.system_dynamic_model
        if system_dynamics is None:
            raise RuntimeError(
                "RFImagineEnvWrapper.system_dynamic_model is None. "
                "Call `set_system_dynamics()` on the wrapper before attaching it to WorldModelFKAction."
            )
        self.system_dynamics = system_dynamics

        # Dimensionalities for world model input.
        dim_params = env_wrapper.dim_params
        self._dynamic_dim = int(dim_params["dynamic_dim"])
        self._world_action_dim = int(dim_params["action_dim"])
        self._history_horizon = int(self.system_dynamics.history_horizon)

        # Initialize history buffers with zeros (start from all-zero dynamics).
        self._dynamic_history = torch.zeros(
            self.num_envs, self._history_horizon, self._dynamic_dim, device=self.device
        )
        self._action_history = torch.zeros(
            self.num_envs, self._history_horizon, self._world_action_dim, device=self.device
        )

        self._counter = 0
        self._last_state_cache = None

    # --------------------------------------------------------------------- #
    # Operations
    # --------------------------------------------------------------------- #

    def process_actions(self, actions: torch.Tensor) -> None:
        """Store the raw high-level actions from the policy."""
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected high-level action dim {self.action_dim}, "
                f"but got {actions.shape[-1]}."
            )
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        """Apply actions by rolling out the world model and writing FK state.

        The behavior is governed by the decimation factor:

        - On steps where ``step_idx % decimation == 0``, the world model is rolled out
          using the current high-level actions and the maintained history.
        - On intermediate steps, the last predicted state is re-applied without
          invoking the world model again.

        In both cases, only environment **buffers** are updated. The actual write to the
        simulator is performed later by Isaac Lab's ``scene.write_data_to_sim()``.
        """
        if self.system_dynamics is None or self._env_wrapper is None:
            # World model FK is not enabled; do nothing.
            return

        # Determine whether to perform a world-model rollout at this step.
        if self._counter % self.cfg.decimation == 0:
            # Full world-model step.
            parsed_state, *_ = self._step_world_model(self.processed_actions)
            self._apply_parsed_state_to_env(parsed_state)
            self._counter = 0
        else:
            # Re-apply last predicted state (if available).
            if self._last_state_cache is not None:
                self._apply_parsed_state_to_env(self._last_state_cache)

        self._counter += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset internal histories on environment reset."""
        if env_ids is None:
            self._raw_actions.zero_()
            if self._dynamic_history is not None:
                self._dynamic_history.zero_()
            if self._action_history is not None:
                self._action_history.zero_()
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            self._raw_actions[env_ids_t] = 0.0
            if self._dynamic_history is not None:
                self._dynamic_history[env_ids_t] = 0.0
            if self._action_history is not None:
                self._action_history[env_ids_t] = 0.0

        self._counter = 0
        self._last_state_cache = None

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _step_world_model(
        self, actions: torch.Tensor
    ) -> tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run a single world-model step and update internal histories.

        Args:
            actions: High-level actions from the policy, shape (num_envs, action_dim).

        Returns:
            parsed_state: Dict of parsed dynamic components (e.g., root pose, joint states).
            reward_pred: Optional predicted reward tensor.
            term_pred: Optional predicted termination logits/tensor.
        """
        assert self.system_dynamics is not None
        assert self._dynamic_history is not None
        assert self._action_history is not None
        assert self._dynamic_dim is not None
        assert self._world_action_dim is not None
        assert self._history_horizon is not None

        # For now we assume that the high-level action has the same dimension as the
        # world model action dimension. If not, users should provide a custom mapping.
        if actions.shape[-1] != self._world_action_dim:
            raise ValueError(
                f"World model expects action_dim={self._world_action_dim}, "
                f"but received actions of dim={actions.shape[-1]}."
            )

        # Construct dynamic and action sequences for the world model.
        # We append the current action to the existing history.
        current_action = actions.unsqueeze(1)  # [N, 1, action_dim]
        dynamic_history = self._dynamic_history
        action_history = self._action_history

        # As in imagination_step: concatenate current state/action to history.
        # Note: for dynamics we use the last frame of dynamic_history as current.
        current_dynamic = dynamic_history[:, -1:]  # [N, 1, dynamic_dim]
        dynamic_seq = torch.cat([dynamic_history, current_dynamic], dim=1)
        action_seq = torch.cat([action_history, current_action], dim=1)

        with torch.no_grad():
            next_dynamic, ext_pred, contact_pred, term_pred, reward_pred = self.system_dynamics(
                dynamic_seq, action_seq
            )

        # Update internal histories: shift and append.
        self._dynamic_history = torch.cat(
            [dynamic_history[:, 1:], next_dynamic.unsqueeze(1)], dim=1
        )
        self._action_history = torch.cat(
            [action_history[:, 1:], current_action], dim=1
        )

        # Parse flat dynamic vector into structured state.
        parsed_state = self._parse_dynamic(next_dynamic)

        # Cache last state for decimation reuse.
        self._last_state_cache = parsed_state

        return parsed_state, reward_pred, term_pred

    def _parse_dynamic(self, dynamic: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse flat dynamic vector into structured robot state.

        By default, this method assumes the following layout:

        .. code-block:: text

            dynamic = [
                root_pos(3),
                root_quat(4),
                root_lin_vel(3),
                root_ang_vel(3),
                joint_pos(num_joints),
                joint_vel(num_joints),
            ]

        This corresponds to a total dimension of ``13 + 2 * num_joints``.

        If the actual layout differs, users should subclass :class:`WorldModelFKAction`
        and override this method to provide a task/robot-specific parsing.
        """
        num_envs, dyn_dim = dynamic.shape
        num_joints = self.robot.num_joints
        expected_dim = 13 + 2 * num_joints
        if dyn_dim != expected_dim:
            raise ValueError(
                f"World model dynamic_dim={dyn_dim} does not match expected "
                f"layout 13 + 2 * num_joints={expected_dim}. "
                "Override `_parse_dynamic` for your specific layout."
            )

        root_pos = dynamic[:, 0:3]
        root_quat = dynamic[:, 3:7]
        root_lin_vel = dynamic[:, 7:10]
        root_ang_vel = dynamic[:, 10:13]
        joint_pos = dynamic[:, 13 : 13 + num_joints]
        joint_vel = dynamic[:, 13 + num_joints : 13 + 2 * num_joints]

        # Normalize quaternion to avoid numerical drift.
        root_quat = root_quat / (root_quat.norm(dim=-1, keepdim=True) + 1e-8)

        return {
            "root_pos": root_pos,
            "root_quat": root_quat,
            "root_lin_vel": root_lin_vel,
            "root_ang_vel": root_ang_vel,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }

    def _apply_parsed_state_to_env(self, parsed_state: Dict[str, torch.Tensor]) -> None:
        """Write parsed state into the underlying environment's state buffers.

        This base implementation is intentionally conservative and does **not**
        directly modify any buffers, because the exact mapping between parsed
        state and the simulator buffers (e.g. ``default_root_state``,
        ``root_state_w``, joint buffers, etc.) is task- and robot-specific.

        Users are expected to either:

        - Override this method in a subclass tailored to their Isaac Lab task, or
        - Contribute a concrete implementation once a standard layout for the
          target robot family is established.

        A typical implementation would:

        - Update root state buffers with ``root_pos``, ``root_quat``,
          ``root_lin_vel``, ``root_ang_vel``.
        - Update joint position/velocity buffers with ``joint_pos``, ``joint_vel``.
        - Rely on Isaac Lab's ``scene.write_data_to_sim()`` to push these buffers
          to the simulator on each physics step.
        """
        # Example skeleton (commented out):
        #
        # task = self.env
        # robot = self.robot
        # TODO set robot with in origin forward
        # self.env.scene.env_origins
        # root_states = self.robot.data.default_root_state + self.env.scene.env_origins
        # root_states[:, :3] = parsed_state["root_pos"]
        # root_states[:, 3:7] = parsed_state["root_quat"]
        # root_states[:, 7:10] = parsed_state["root_lin_vel"]
        # root_states[:, 10:] = parsed_state["root_ang_vel"]
        # joint_pos = robot.data.joint_pos
        # joint_vel = robot.data.joint_vel
        # joint_pos[:, :] = parsed_state["joint_pos"]
        # joint_vel[:, :] = parsed_state["joint_vel"]
        #
        # For safety, we leave this as a no-op by default.
        _ = parsed_state  # silence unused-variable warning
        return


@configclass
class WorldModelFKActionCfg(ActionTermCfg):
    """Configuration for :class:`WorldModelFKAction`.

    Parameters
    ----------
    asset_name:
        Name of the Articulation asset in the Isaac Lab scene that this action
        term controls.
    action_dim:
        Dimension of the high-level action coming from the RL policy.
    decimation:
        Decimation factor for world-model rollouts. A value of ``d`` means that
        the world model is invoked every ``d`` calls to :meth:`apply_actions`,
        and the last predicted state is reused on intermediate steps.
    """

    class_type: type[ActionTerm] = WorldModelFKAction
    asset_name: str = MISSING
    action_dim: int = MISSING
    decimation: int = 1

