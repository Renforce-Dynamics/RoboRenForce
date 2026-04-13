from __future__ import annotations

from typing import Optional, Dict, Any, List

import numpy as np
import torch

from .exploration_coefficient_epo import EPOExplorationCoefficient


class EPOObserver:
    """
    EPO-style observer for block-level evolution inside a single policy.

    Follows the design of reference/EPO's EPOObserver:
    - Aggregate true_objective (or reward) per block;
    - Periodically select best blocks and overwrite/merge parameters of the worst block.
    """

    def __init__(
        self,
        expl_coef: EPOExplorationCoefficient,
        num_envs: int,
        interval_steps: int = 20_000_000,
        warmup_steps: int = 200_000_000,
        best_embeddings: bool = True,
    ):
        self.expl_coef = expl_coef
        self.num_envs = num_envs
        self.interval_steps = interval_steps
        self.warmup_steps = warmup_steps
        self.best_embeddings = best_embeddings

        # derived from exploration coefficient
        self.num_blocks = expl_coef.num_blocks
        self.block_size = expl_coef.block_size

        # tracking
        self._reset_iteration_state()

        # current global frame, injected from runner
        self.frame: int = 0

    def _reset_iteration_state(self):
        self.finished_envs: set[int] = set()
        self.last_objectives: Optional[List[float]] = None
        self.curr_objective_per_block: Optional[List[float]] = None
        self.best_objective_per_block: Optional[List[float]] = None
        self.target_objective_known: bool = False
        # current EPO iteration index (frame // interval_steps)
        self.epo_iteration: int = -1

    # ------------------------------------------------------------------ #
    # Public API used by runner
    # ------------------------------------------------------------------ #

    def update_frame(self, frame: int) -> None:
        """Update current global frame from runner."""
        self.frame = int(frame)

    def process_infos(
        self,
        infos: Dict[str, Any],
        done_indices: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Consume env infos at each step to accumulate per-block objectives.

        Args:
            infos: info dict returned by env.step(...)
            done_indices: indices of finished envs (same semantics as in reference/EPO)
            rewards: reward tensor from env.step(...), used as fallback if true_objective is not available
        """
        if not self.best_embeddings:
            return

        if not isinstance(infos, dict):
            return

        # Determine which objective to use: true_objective if available, otherwise reward
        if "true_objective" in infos:
            objective = infos["true_objective"]
        elif rewards is not None:
            # Fallback to reward if true_objective is not provided
            objective = rewards
        else:
            # No objective available, skip
            return

        # Lazy init on first call
        if self.last_objectives is None:
            num_valid_envs = objective.shape[0]
            self.last_objectives = [-1e9] * num_valid_envs
            self.curr_objective_per_block = [-1e9] * self.num_blocks
            self.best_objective_per_block = [-1e9] * self.num_blocks

        assert objective.shape[0] % self.num_blocks == 0, f"objective size ({objective.shape[0]}) must be divisible by num_blocks ({self.num_blocks})"

        done_list = done_indices.view(-1).tolist()
        for idx in done_list:
            if idx < 0 or idx >= len(self.last_objectives):
                continue
            self.finished_envs.add(idx)
            # Use cumulative reward for done episodes if using reward fallback
            # For true_objective, it's typically already per-episode
            self.last_objectives[idx] = float(objective[idx].item())

        # Once all envs have finished at least one episode, compute per-block mean objective
        if len(self.finished_envs) >= len(self.last_objectives):
            self.target_objective_known = True
            all_vals = np.array(self.last_objectives, dtype=np.float32)
            bsize = len(self.last_objectives) // self.num_blocks
            for b in range(self.num_blocks):
                start = b * bsize
                end = (b + 1) * bsize
                self.curr_objective_per_block[b] = float(all_vals[start:end].mean())

            # Track the best objective per block for current EPO iteration
            for b in range(self.num_blocks):
                if self.curr_objective_per_block[b] > self.best_objective_per_block[b]:
                    self.best_objective_per_block[b] = self.curr_objective_per_block[b]

    def after_steps(self) -> None:
        """
        Called periodically from runner (e.g. after each training iteration),
        decides whether to trigger a block-level merge.
        """
        if not self.best_embeddings:
            return

        env_frames = self.frame

        # Initialize epo_iteration on first call
        if self.epo_iteration == -1:
            self.epo_iteration = env_frames // self.interval_steps

        iteration = env_frames // self.interval_steps
        if iteration <= self.epo_iteration:
            return

        if not self.target_objective_known:
            # Not enough data yet to compute reliable objectives
            return

        if env_frames < self.warmup_steps:
            # Give policy more time to adapt before modifying parameters
            return

        # Select best and worst blocks according to best_objective_per_block
        objs = np.array(self.best_objective_per_block, dtype=np.float32)
        valid_mask = objs > -1e8
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return

        valid_objs = objs[valid_indices]
        order = np.argsort(-valid_objs)
        sorted_blocks = valid_indices[order]

        # Take first 2 as best_blocks and last 1 as worst_block
        best_blocks = sorted_blocks[:2].tolist()
        worst_block = int(sorted_blocks[-1])

        # Call ExplorationCoefficient.merge_block_params to update target block
        self.expl_coef.merge_block_params(best_blocks=best_blocks, target_block=worst_block, mode="copy_best")

        # Reset iteration state and move to the next EPO iteration
        self.best_objective_per_block = [-1e9] * self.num_blocks
        self.curr_objective_per_block = [-1e9] * self.num_blocks
        self.finished_envs.clear()
        self.target_objective_known = False
        self.epo_iteration = iteration

