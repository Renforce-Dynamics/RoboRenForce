"""
MLP Baseline Policy — Simple feedforward policy for testing and debugging.

No pretrained model, no VLM, no heavy dependencies.
Useful for verifying the training loop works end-to-end.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from RoboRenForce.prototype.embodied import BasePolicy, ForwardType
from RRF_models.modules.value_head import ValueHead


class MLPBaselinePolicy(BasePolicy):
    """Minimal MLP policy: flatten obs → MLP → actions.

    Only uses proprioception (states). Ignores images and text.
    Good for sanity-checking RL loops before plugging in a real VLM.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_horizon: int = 1,
        hidden_dims: tuple[int, ...] = (256, 256),
        use_value_head: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Actor MLP
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.GELU()])
            prev = h
        layers.append(nn.Linear(prev, action_dim * action_horizon))
        self.actor_mlp = nn.Sequential(*layers)

        # Optional value head
        self.value_head: Optional[ValueHead] = None
        if use_value_head:
            self.value_head = ValueHead(input_dim=state_dim, hidden_dims=(256, 64))

    def predict_action(self, obs: dict[str, Any], **kwargs) -> torch.Tensor:
        states = obs["states"]
        with torch.no_grad():
            raw = self.actor_mlp(states)
        return raw.view(states.shape[0], self.action_horizon, self.action_dim)

    def forward(
        self,
        forward_type: ForwardType = ForwardType.PRETRAIN,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if forward_type == ForwardType.INFERENCE:
            return {"actions": self.predict_action(kwargs["obs"])}
        elif forward_type == ForwardType.PRETRAIN:
            return self.pretrain_forward(**kwargs)
        elif forward_type == ForwardType.PPO:
            return self.ppo_forward(**kwargs)
        else:
            raise NotImplementedError(f"MLPBaselinePolicy: {forward_type} not implemented")

    def pretrain_forward(
        self,
        obs: dict[str, Any],
        target_actions: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        states = obs["states"]
        raw = self.actor_mlp(states)
        pred = raw.view(states.shape[0], self.action_horizon, self.action_dim)
        loss = nn.functional.mse_loss(pred, target_actions)
        return {"loss": loss, "action_loss": loss, "pred_actions": pred}

    def ppo_forward(
        self,
        obs: dict[str, Any],
        actions: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        states = obs["states"]
        raw = self.actor_mlp(states)
        pred = raw.view(states.shape[0], self.action_horizon, self.action_dim)

        action_diff = actions - pred
        logprobs = -0.5 * (action_diff ** 2).sum(dim=-1).mean(dim=-1)

        result = {"logprobs": logprobs, "pred_actions": pred}
        if self.value_head is not None:
            result["values"] = self.value_head(states)
        return result

    def get_value(self, obs: dict[str, Any]) -> Optional[torch.Tensor]:
        if self.value_head is None:
            return None
        return self.value_head(obs["states"])
