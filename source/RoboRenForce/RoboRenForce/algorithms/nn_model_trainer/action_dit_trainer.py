"""Trainer for FlowMatchingActionDiT.

Handles optimization, mixed precision, gradient clipping, and LR scheduling
for offline supervised training of the action DiT head.
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from RoboRenForce import configclass
from RoboRenForce.utils.template import ClassTemplateBaseCfg


class ActionDiTTrainer:
    """Trainer for FlowMatchingActionDiT.

    Operates on batches of (vl_embs, actions, action_mask, state).
    Supports mixed precision (bf16), gradient clipping, and cosine LR schedule.
    """

    cfg: "ActionDiTTrainerCfg"

    def __init__(
        self,
        cfg: "ActionDiTTrainerCfg",
        action_model: nn.Module,
    ) -> None:
        self.cfg = cfg
        self.action_model = action_model

        # Optimizer
        self.optimizer = optim.AdamW(
            self.action_model.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
        )

        # Grad scaler for mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

        # LR scheduler (built lazily when total_steps is known)
        self._scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self._completed_steps = 0

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def build_scheduler(self, total_steps: int):
        """Build cosine LR schedule with linear warmup."""
        warmup = self.cfg.num_warmup_steps
        min_lr_ratio = self.cfg.min_lr / self.cfg.learning_rate

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self._scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        # Advance scheduler to current step if resuming
        for _ in range(self._completed_steps):
            self._scheduler.step()

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def train_step(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Run one training step.

        Args:
            vl_embs:     (B, S, D)          conditioning embeddings
            actions:     (B, T, action_dim)  ground-truth actions
            action_mask: (B, T, action_dim)  valid-action mask
            state:       (B, state_dim)      optional proprioceptive state

        Returns:
            Dict with 'action_loss' and 'lr'.
        """
        cfg = self.cfg
        self.action_model.train()

        # Optionally repeat for multiple diffusion samples per batch
        if cfg.repeated_diffusion_steps > 1:
            r = cfg.repeated_diffusion_steps
            vl_embs = vl_embs.repeat(r, 1, 1)
            actions = actions.repeat(r, 1, 1)
            action_mask = action_mask.repeat(r, 1, 1)
            if state is not None:
                state = state.repeat(r, 1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_amp):
            loss = self.action_model(vl_embs, actions, action_mask, state)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        if cfg.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.action_model.parameters(), cfg.max_grad_norm
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self._scheduler is not None:
            self._scheduler.step()

        self._completed_steps += 1

        return {
            "action_loss": loss.item(),
            "lr": self.current_lr,
        }

    # ------------------------------------------------------------------
    # Batch update (for replay-buffer / generator interface)
    # ------------------------------------------------------------------

    def update(
        self,
        data_iter: Iterator,
        num_steps: int = 1,
    ) -> Dict[str, float]:
        """Run multiple training steps from a data iterator.

        Each item from data_iter should be a dict with keys:
            vl_embs, actions, action_mask, state (optional)

        Returns averaged losses.
        """
        accum = {}
        count = 0

        for _ in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            stats = self.train_step(
                vl_embs=batch["vl_embs"],
                actions=batch["actions"],
                action_mask=batch["action_mask"],
                state=batch.get("state"),
            )

            for k, v in stats.items():
                accum[k] = accum.get(k, 0.0) + v
            count += 1

        if count == 0:
            return {"action_loss": 0.0, "lr": self.current_lr}

        return {k: v / count for k, v in accum.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        data_iter: Iterator,
        num_steps: int = 1,
    ) -> Dict[str, float]:
        """Evaluate on a data iterator. Returns averaged metrics."""
        self.action_model.eval()
        accum_loss = 0.0
        count = 0

        for _ in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.use_amp):
                loss = self.action_model(
                    batch["vl_embs"],
                    batch["actions"],
                    batch["action_mask"],
                    batch.get("state"),
                )
            accum_loss += loss.item()
            count += 1

        return {"eval_action_loss": accum_loss / max(1, count)}

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def state_dict(self):
        sd = {
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "completed_steps": self._completed_steps,
        }
        if self._scheduler is not None:
            sd["scheduler"] = self._scheduler.state_dict()
        return sd

    def load_state_dict(self, sd):
        self.optimizer.load_state_dict(sd["optimizer"])
        self.scaler.load_state_dict(sd["scaler"])
        self._completed_steps = sd.get("completed_steps", 0)
        if self._scheduler is not None and "scheduler" in sd:
            self._scheduler.load_state_dict(sd["scheduler"])


@configclass
class ActionDiTTrainerCfg(ClassTemplateBaseCfg):
    class_type: type[ActionDiTTrainer] = ActionDiTTrainer

    # Optimizer
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 1e-8

    # LR schedule
    num_warmup_steps: int = 1000
    min_lr: float = 5e-7

    # Training
    max_grad_norm: float = 1.0
    use_amp: bool = True
    repeated_diffusion_steps: int = 4

    def construct_from_cfg(self, *args, **kwargs) -> "ActionDiTTrainer":
        return super().construct_from_cfg(*args, **kwargs)
