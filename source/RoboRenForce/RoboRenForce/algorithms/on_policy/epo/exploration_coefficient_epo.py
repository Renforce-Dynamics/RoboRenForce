from __future__ import annotations

from typing import List

import torch

from RoboRenForce import configclass
from RoboRenForce.algorithms.on_policy.sapg.exploration_coefficient import (
    ExplorationCoefficient as SAPGExplorationCoefficient,
    ExplorationCoefficientCfg as SAPGExplorationCoefficientCfg,
)
from RoboRenForce.utils.template.module_base import ModuleBaseCfg


class EPOExplorationCoefficient(SAPGExplorationCoefficient):
    """
    EPO-specialized exploration coefficient manager.

    Extends SAPG ExplorationCoefficient by adding a block-level merge interface
    that can be used by EPOObserver to evolve block parameters within a single policy.
    """

    def merge_block_params(
        self,
        best_blocks: List[int],
        target_block: int,
        mode: str = "copy_best",
    ) -> None:
        """
        Merge exploration-related parameters from best blocks into a target block.

        Args:
            best_blocks: Indices of best-performing blocks.
            target_block: Index of the block to overwrite (worst block).
            mode: Merge strategy:
                - "copy_best": copy the first best block;
                - "mean": average all best blocks.
        """
        if len(best_blocks) == 0:
            return

        # Helper to slice per-block views from a 1D or 2D tensor
        def _get_block_slice(t: torch.Tensor, block_idx: int) -> torch.Tensor:
            start = block_idx * self.block_size
            end = (block_idx + 1) * self.block_size
            return t[start:end]

        # Collect embeddings and reward coefficients of best blocks
        best_embds = []
        best_reward_coefs = []
        for b in best_blocks:
            best_embds.append(_get_block_slice(self.coef_embd, b))
            best_reward_coefs.append(_get_block_slice(self.reward_coef, b))

        if mode == "copy_best":
            merged_embd = best_embds[0]
            merged_reward_coef = best_reward_coefs[0]
        elif mode == "mean":
            merged_embd = torch.stack(best_embds, dim=0).mean(dim=0)
            merged_reward_coef = torch.stack(best_reward_coefs, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown merge mode: {mode}")

        # Write back into target block
        start = target_block * self.block_size
        end = (target_block + 1) * self.block_size
        self.coef_embd[start:end] = merged_embd
        self.reward_coef[start:end] = merged_reward_coef


@configclass
class EPOExplorationCoefficientCfg(SAPGExplorationCoefficientCfg):
    """
    Config for EPOExplorationCoefficient.

    Only overrides class_type to point to EPOExplorationCoefficient; all other
    fields are inherited from the SAPG configuration.
    """

    class_type: type[EPOExplorationCoefficient] = EPOExplorationCoefficient

