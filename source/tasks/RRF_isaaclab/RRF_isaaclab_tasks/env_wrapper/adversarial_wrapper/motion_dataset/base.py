from __future__ import annotations

import os
from dataclasses import MISSING
from typing import List

import numpy as np
import torch

from RoboRenForce import configclass
from RoboRenForce.utils.math import quat_apply_inverse, quat_conjugate, quat_apply


class MotionDataset:
    """AMP motion dataset built from multiple motion files.

    Each motion file is expected to contain numpy arrays with the following keys:
        - ``joint_pos`` (T, dof)
        - ``joint_vel`` (T, dof)
        - ``body_pos_w`` (T, B, 3)
        - ``body_quat_w`` (T, B, 4)
        - ``body_lin_vel_w`` (T, B, 3)
        - ``body_ang_vel_w`` (T, B, 3)
        - ``fps`` (scalar)

    The dataset pre-builds valid (t, t+1) transition indices for efficient sampling
    of (state, next_state) pairs.
    """

    def __init__(
        self,
        cfg: "MotionDatasetCfg",
        env,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.robot = env.scene[cfg.asset_name]
        self.motion_files = cfg.motion_files
        self.observation_terms = cfg.amp_obs_terms

        body_names = cfg.body_names
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=device,
        )

        anchor_name = cfg.anchor_name
        self.anchor_index = torch.tensor(
            self.robot.find_bodies(anchor_name, preserve_order=True)[0],
            dtype=torch.long,
            device=device,
        )

        self.load_motions()
        self.init_observation_dims()

    # ------------------------------------------------------------------ #
    # Loading and properties
    # ------------------------------------------------------------------ #
    def load_motions(self):
        """Load and concatenate all motion files into contiguous tensors."""
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        fps_list = []
        traj_lengths = []

        for f in self.motion_files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"Invalid motion file: {f}")
            data = np.load(f)

            fps_list.append(float(np.asarray(data["fps"]).reshape(-1)[0]))
            traj_len = data["joint_pos"].shape[0]
            traj_lengths.append(traj_len)

            joint_pos_list.append(torch.tensor(data["joint_pos"], dtype=torch.float32))
            joint_vel_list.append(torch.tensor(data["joint_vel"], dtype=torch.float32))
            body_pos_w_list.append(torch.tensor(data["body_pos_w"], dtype=torch.float32))
            body_quat_w_list.append(torch.tensor(data["body_quat_w"], dtype=torch.float32))
            body_lin_vel_w_list.append(
                torch.tensor(data["body_lin_vel_w"], dtype=torch.float32)
            )
            body_ang_vel_w_list.append(
                torch.tensor(data["body_ang_vel_w"], dtype=torch.float32)
            )

        self.joint_pos = torch.cat(joint_pos_list, dim=0).to(self.device)
        self.joint_vel = torch.cat(joint_vel_list, dim=0).to(self.device)
        self.body_pos_w_all = torch.cat(body_pos_w_list, dim=0).to(self.device)
        self.body_quat_w_all = torch.cat(body_quat_w_list, dim=0).to(self.device)
        self.body_lin_vel_w_all = torch.cat(body_lin_vel_w_list, dim=0).to(self.device)
        self.body_ang_vel_w_all = torch.cat(body_ang_vel_w_list, dim=0).to(self.device)

        self.total_dataset_size = sum(traj_lengths)
        self._traj_lengths = traj_lengths
        self.fps_list = fps_list

        self.index_t, self.index_tp1 = self._build_transition_indices(
            traj_lengths, self.device
        )

    # Basic index helpers ------------------------------------------------
    def subtract_flaten(self, target: torch.Tensor):
        target = target[:, self.body_indexes]
        return target.reshape(self.total_dataset_size, -1)

    @property
    def body_pos_w(self):
        return self.body_pos_w_all[:, self.body_indexes].reshape(
            self.total_dataset_size, -1
        )

    @property
    def body_quat_w(self):
        return self.body_quat_w_all[:, self.body_indexes].reshape(
            self.total_dataset_size, -1
        )

    @property
    def body_lin_vel_w(self):
        return self.body_lin_vel_w_all[:, self.body_indexes].reshape(
            self.total_dataset_size, -1
        )

    @property
    def body_ang_vel_w(self):
        return self.body_ang_vel_w_all[:, self.body_indexes].reshape(
            self.total_dataset_size, -1
        )

    # Anchor-based quantities -------------------------------------------
    @property
    def anchor_pos_w(self):
        return self.body_pos_w_all[:, self.anchor_index].reshape(
            self.total_dataset_size, -1
        )

    @property
    def anchor_quat_w(self):
        return self.body_quat_w_all[:, self.anchor_index].reshape(
            self.total_dataset_size, -1
        )

    @property
    def anchor_lin_vel_w(self):
        return self.body_lin_vel_w_all[:, self.anchor_index].reshape(
            self.total_dataset_size, -1
        )

    @property
    def anchor_ang_vel_w(self):
        return self.body_ang_vel_w_all[:, self.anchor_index].reshape(
            self.total_dataset_size, -1
        )

    @property
    def anchor_height(self):
        return self.anchor_pos_w[:, -1]

    # Body quantities in anchor frame -----------------------------------
    @property
    def _anchor_pos(self):
        return self.anchor_pos_w.view(self.total_dataset_size, -1, 3)

    @property
    def _anchor_quat(self):
        return self.anchor_quat_w.view(self.total_dataset_size, -1, 4)

    @property
    def body_pos_b(self):
        """Body positions expressed in anchor-local frame."""
        pos_w = self.body_pos_w_all[:, self.body_indexes]
        anchor_pos = self._anchor_pos.unsqueeze(1)
        anchor_quat = self._anchor_quat.unsqueeze(1)

        rel = pos_w - anchor_pos
        rel_local = quat_apply_inverse(anchor_quat, rel)
        return rel_local.reshape(self.total_dataset_size, -1)

    @property
    def body_quat_b(self):
        """Body orientations expressed in anchor-local frame."""
        q_body = self.body_quat_w_all[:, self.body_indexes]
        q_anchor = self._anchor_quat.unsqueeze(1)

        q_anchor_inv = quat_conjugate(q_anchor)
        q_rel = quat_apply(q_anchor_inv, q_body)
        return q_rel.reshape(self.total_dataset_size, -1)

    @property
    def body_lin_vel_b(self):
        """Body linear velocities in anchor-local frame."""
        v_body = self.body_lin_vel_w_all[:, self.body_indexes]
        v_anchor = self.anchor_lin_vel_w.view(self.total_dataset_size, 1, 3)
        rel = v_body - v_anchor
        rel_local = quat_apply_inverse(self._anchor_quat.unsqueeze(1), rel)
        return rel_local.reshape(self.total_dataset_size, -1)

    @property
    def body_ang_vel_b(self):
        """Body angular velocities in anchor-local frame."""
        w_body = self.body_ang_vel_w_all[:, self.body_indexes]
        w_anchor = self.anchor_ang_vel_w.view(self.total_dataset_size, 1, 3)
        rel = w_body - w_anchor
        rel_local = quat_apply_inverse(self._anchor_quat.unsqueeze(1), rel)
        return rel_local.reshape(self.total_dataset_size, -1)

    @property
    def base_lin_vel(self):
        """Base (anchor) linear velocity expressed in base frame."""
        v_w = self.anchor_lin_vel_w
        q_w = self.anchor_quat_w
        v_b = quat_apply_inverse(q_w, v_w)
        return v_b

    @property
    def base_ang_vel(self):
        """Base (anchor) angular velocity expressed in base frame."""
        w_w = self.anchor_ang_vel_w
        q_w = self.anchor_quat_w
        w_b = quat_apply_inverse(q_w, w_w)
        return w_b

    # ------------------------------------------------------------------ #
    # Observation term utilities
    # ------------------------------------------------------------------ #
    def observation_dim_cast(self, name) -> int:
        """Return flattened dimension of an observation term."""
        if hasattr(self, name):
            obs_term: torch.Tensor = getattr(self, name)
            if not isinstance(obs_term, torch.Tensor):
                raise TypeError(f"Invalid observation term: {name}")
            return obs_term.shape[-1]
        raise NotImplementedError(f"Unknown observation term: {name}")

    def init_observation_dims(self):
        """Compute observation dimension for all configured AMP observation terms."""
        observation_dims = []
        for obs_term in self.observation_terms:
            observation_dims.append(self.observation_dim_cast(obs_term))
        self.observation_dim = sum(observation_dims)
        self.observation_dims = observation_dims

    # ------------------------------------------------------------------ #
    # Transition indices and sampling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_transition_indices(traj_lengths: List[int], device: str):
        """Build valid (t, t+1) pairs without crossing trajectory boundaries."""
        idx_t = []
        idx_tp1 = []

        offset = 0
        for L in traj_lengths:
            if L < 2:
                offset += L
                continue
            t = torch.arange(offset, offset + L - 1)
            idx_t.append(t)
            idx_tp1.append(t + 1)
            offset += L

        idx_t = torch.cat(idx_t).to(device)
        idx_tp1 = torch.cat(idx_tp1).to(device)
        return idx_t, idx_tp1

    def sample_batch(self, batch_size: int):
        """Sample a batch of transition indices (t, t+1)."""
        idx = torch.randint(0, len(self.index_t), (batch_size,), device=self.device)
        t = self.index_t[idx]
        tp1 = self.index_tp1[idx]
        return t, tp1

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Feed-forward generator yielding (state_t, state_t+1) AMP observations."""
        for _ in range(num_mini_batch):
            t, tp1 = self.sample_batch(mini_batch_size)
            res_t, res_tp1 = self.build_transition(t, tp1)
            yield res_t, res_tp1

    def build_transition(self, t, tp1):
        """Build (s_t, s_{t+1}) concatenating all AMP observation terms."""
        res_t, res_tp1 = [], []
        for term in self.observation_terms:
            _t, _tp1 = getattr(self, term)[t], getattr(self, term)[tp1]
            res_t.append(_t)
            res_tp1.append(_tp1)
        res_t = torch.cat(res_t, dim=-1)
        res_tp1 = torch.cat(res_tp1, dim=-1)
        return res_t, res_tp1


@configclass
class MotionDatasetCfg:
    """Configuration for AMP MotionDataset."""

    class_type: type[MotionDataset] = MotionDataset
    asset_name: str = "robot"
    motion_files: List[str] = MISSING
    body_names: List[str] = MISSING
    amp_obs_terms: List[str] = MISSING
    anchor_name: str = MISSING


__all__ = ["MotionDataset", "MotionDatasetCfg"]

