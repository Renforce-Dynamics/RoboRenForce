from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import MISSING
from typing import Optional, Dict

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBaseCfg
from RoboRenForce.components.nn_models.system_dynamics.system_dynamics_base import (
    SystemDynamicsBase,
    SystemDynamicsBaseCfg,
)
from RoboRenForce.networks.transformer import (
    TransformerBackbone,
    TransformerBackboneCfg,
)


class SystemDynamicsTransformer(SystemDynamicsBase):
    """
    Transformer-based system dynamics model for MBPO-style algorithms.

    This model replaces the MLP trunk with a generic Transformer backbone.
    It uses a history of (dynamic, action) pairs per transition and lets
    the Transformer attend over this short sequence, using the last token
    representation as the summary for predicting the next state and
    optional auxiliary signals.
    """

    cfg: "SystemDynamicsTransformerCfg"

    def __init__(self, cfg: "SystemDynamicsTransformerCfg", dim_params: Dict[str, int], device: str = "cpu"):
        # Infer dimensions from env dim_params + cfg overrides.
        dynamic_dim = dim_params.get("dynamic_dim")
        action_dim = dim_params.get("action_dim")
        extension_dim = dim_params.get("extension_dim", 0)
        contact_dim = dim_params.get("contact_dim", 0)
        termination_dim = dim_params.get("termination_dim", 0)
        reward_dim = dim_params.get("reward_dim", 1)
        history_horizon = cfg.history_horizon

        super().__init__(
            state_dim=dynamic_dim,
            action_dim=action_dim,
            extension_dim=extension_dim,
            contact_dim=contact_dim,
            termination_dim=termination_dim,
            reward_dim=reward_dim,
            history_horizon=history_horizon,
            device=device,
        )

        self.cfg = cfg
        self.dynamic_dim = self.state_dim

        # Each token is a concatenation of (dynamic_t, action_t)
        token_input_dim = self.dynamic_dim + self.action_dim
        model_dim = cfg.backbone_cfg.dim

        # 1) Input projection: (dynamic, action) -> token embedding
        self.input_proj = nn.Linear(token_input_dim, model_dim)

        # 2) Transformer backbone operating on sequences of tokens of length = history_horizon
        self.backbone: TransformerBackbone = cfg.backbone_cfg.class_type(cfg.backbone_cfg)

        # 3) Prediction heads based on the final (last-token) hidden state
        last_dim = model_dim
        self.dynamic_head = nn.Linear(last_dim, self.dynamic_dim)
        self.extension_head = (
            nn.Linear(last_dim, self.extension_dim) if self.extension_dim > 0 else None
        )
        self.contact_head = (
            nn.Linear(last_dim, self.contact_dim) if self.contact_dim > 0 else None
        )
        self.termination_head = (
            nn.Linear(last_dim, self.termination_dim) if self.termination_dim > 0 else None
        )
        self.reward_head = (
            nn.Linear(last_dim, self.reward_dim) if self.reward_dim > 0 else None
        )

        self.to(device)

    def _encode_history(
        self,
        dynamic_window: torch.Tensor,
        action_window: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a history window of dynamics and actions with the Transformer backbone.

        Args:
            dynamic_window: Tensor of shape [B, H, dynamic_dim]
            action_window: Tensor of shape [B, H, action_dim]

        Returns:
            feat_last: Tensor of shape [B, model_dim] corresponding to the last token.
        """
        tokens = torch.cat([dynamic_window, action_window], dim=-1)  # [B, H, dynamic_dim + action_dim]
        tokens = self.input_proj(tokens)  # [B, H, model_dim]

        # For short history windows we typically do not need an explicit mask.
        feat_seq = self.backbone(tokens)  # [B, H, model_dim]
        feat_last = feat_seq[:, -1, :]  # summary for predicting next state
        return feat_last

    def _forward_core(self, dynamic_input: torch.Tensor, action_input: torch.Tensor):
        """
        Core forward for a single-step prediction given a history window.

        Args:
            dynamic_input: [B, H, dynamic_dim]
            action_input: [B, H, action_dim]

        Returns:
            next_dynamic, extension, contact, termination, reward
        """
        feat = self._encode_history(dynamic_input, action_input)
        next_dynamic = self.dynamic_head(feat)
        extension = self.extension_head(feat) if self.extension_head is not None else None
        contact = self.contact_head(feat) if self.contact_head is not None else None
        termination = (
            self.termination_head(feat) if self.termination_head is not None else None
        )
        reward = self.reward_head(feat) if self.reward_head is not None else None
        return next_dynamic, extension, contact, termination, reward

    def forward(self, dynamic_seq: torch.Tensor, action_seq: torch.Tensor):
        """
        Predict next dynamic state using history.

        Args:
            dynamic_seq: [B, T, dynamic_dim] where T >= history_horizon
            action_seq: [B, T, action_dim] where T >= history_horizon

        Returns:
            next_dynamic_pred: [B, dynamic_dim]
            extension_pred: Optional[torch.Tensor]
            contact_pred: Optional[torch.Tensor]
            termination_pred: Optional[torch.Tensor]
            reward_pred: Optional[torch.Tensor]
        """
        B, T, _ = dynamic_seq.shape
        H = self.history_horizon

        if T < H:
            raise ValueError(
                f"Sequence length T={T} is smaller than history_horizon={H} "
                "for SystemDynamicsTransformer.forward."
            )

        dynamic_window = dynamic_seq[:, -H:]  # [B, H, dynamic_dim]
        action_window = action_seq[:, -H:]  # [B, H, action_dim]

        return self._forward_core(dynamic_window, action_window)

    def compute_loss(
        self,
        dynamic_batch: torch.Tensor,
        action_batch: torch.Tensor,
        extension_batch: torch.Tensor | None,
        contact_batch: torch.Tensor | None,
        termination_batch: torch.Tensor | None,
        reward_batch: torch.Tensor | None = None,
    ):
        """
        Compute per-component losses with dual autoregressive mechanism.

        Dual autoregressive mechanism:
        1. State prediction: Uses predicted states for autoregressive prediction (teacher forcing with predictions)
        2. Auxiliary prediction: Uses ground truth states for autoregressive prediction (teacher forcing with ground truth)

        The buffer provides sequences of length `T = history_horizon + forecast_horizon`.
        We predict forecast_horizon steps autoregressively.

        Shapes:
            dynamic_batch: [B, T, dynamic_dim] where T = history_horizon + forecast_horizon
            action_batch: [B, T, action_dim]
            reward_batch: Optional [B, T, reward_dim]
        """
        B, T, _ = dynamic_batch.shape
        H = self.history_horizon
        forecast_horizon = T - H

        if forecast_horizon <= 0:
            raise ValueError(
                f"Sequence length T={T} must be greater than history_horizon={H} "
                "for autoregressive training."
            )

        # Initialize state windows for autoregressive prediction
        # State prediction uses predicted states (autoregressive)
        x_state_batch_pred = dynamic_batch[:, :H].clone()  # [B, H, D_d]
        # Auxiliary prediction uses ground truth states (teacher forcing)
        x_state_batch_gt = dynamic_batch[:, :H].clone()  # [B, H, D_d]

        state_losses = []
        sequence_losses = []
        extension_losses = []
        contact_losses = []
        termination_losses = []
        reward_losses = []

        bce = nn.BCEWithLogitsLoss()

        # Autoregressive prediction loop
        for i in range(forecast_horizon):
            # Get target at step history_horizon + i
            state_target = dynamic_batch[:, H + i]  # [B, D_d]
            
            # Get action window for this step
            # For Transformer, we use the full history window
            action_window = action_batch[:, i + 1:H + i + 1]  # [B, H, D_a]

            # State prediction: use predicted states (autoregressive)
            # This loop focuses on state prediction loss, using predicted states
            next_dynamic_pred, _, _, _, _ = self._forward_core(
                x_state_batch_pred,  # [B, H, D_d] - uses predicted states
                action_window,  # [B, H, D_a]
            )
            state_loss = torch.mean((next_dynamic_pred - state_target) ** 2)
            state_losses.append(state_loss)
            sequence_losses.append(state_loss)  # For Transformer, sequence_loss = state_loss

            # Update state window for next prediction (use predicted state)
            x_state_batch_pred = torch.cat(
                [
                    x_state_batch_pred[:, 1:].clone(),  # [B, H-1, D_d]
                    next_dynamic_pred.unsqueeze(1),  # [B, 1, D_d]
                ],
                dim=1,
            )  # [B, H, D_d]

            # Auxiliary prediction: use ground truth states (teacher forcing)
            # This loop focuses on auxiliary prediction loss, using ground truth states
            _, ext_pred, contact_pred, term_pred, reward_pred = self._forward_core(
                x_state_batch_gt,  # [B, H, D_d] - uses ground truth states
                action_window,  # [B, H, D_a]
            )

            # Compute auxiliary losses
            if self.extension_dim > 0 and extension_batch is not None:
                ext_target = extension_batch[:, H + i]  # [B, extension_dim]
                extension_loss = torch.mean((ext_pred - ext_target) ** 2)
            else:
                extension_loss = torch.tensor(0.0, device=self.device)
            extension_losses.append(extension_loss)

            if self.contact_dim > 0 and contact_batch is not None:
                contact_target = contact_batch[:, H + i]  # [B, contact_dim]
                contact_loss = bce(contact_pred, contact_target)
            else:
                contact_loss = torch.tensor(0.0, device=self.device)
            contact_losses.append(contact_loss)

            if self.termination_dim > 0 and termination_batch is not None:
                term_target = termination_batch[:, H + i]  # [B, termination_dim]
                termination_loss = bce(term_pred, term_target)
            else:
                termination_loss = torch.tensor(0.0, device=self.device)
            termination_losses.append(termination_loss)

            if self.reward_dim > 0 and reward_batch is not None:
                reward_target = reward_batch[:, H + i]  # [B, reward_dim]
                reward_loss = torch.mean((reward_pred - reward_target) ** 2)
            else:
                reward_loss = torch.tensor(0.0, device=self.device)
            reward_losses.append(reward_loss)

            # Update ground truth state window for auxiliary prediction
            x_state_batch_gt = torch.cat(
                [
                    x_state_batch_gt[:, 1:].clone(),  # [B, H-1, D_d]
                    dynamic_batch[:, H + i:H + i + 1],  # [B, 1, D_d] - use ground truth
                ],
                dim=1,
            )  # [B, H, D_d]

        # Average losses over forecast horizon
        dynamic_loss = torch.mean(torch.stack(state_losses))
        sequence_loss = torch.mean(torch.stack(sequence_losses))
        bound_loss = torch.tensor(0.0, device=self.device)
        extension_loss = torch.mean(torch.stack(extension_losses))
        contact_loss = torch.mean(torch.stack(contact_losses))
        termination_loss = torch.mean(torch.stack(termination_losses))
        reward_loss = torch.mean(torch.stack(reward_losses))

        return (
            dynamic_loss,
            sequence_loss,
            bound_loss,
            extension_loss,
            contact_loss,
            termination_loss,
            reward_loss,
        )

    @torch.no_grad()
    def autoregressive_prediction(
        self,
        dynamic_seq: torch.Tensor,
        action_seq: torch.Tensor,
        extension_seq: torch.Tensor | None = None,
        contact_seq: torch.Tensor | None = None,
        termination_seq: torch.Tensor | None = None,
    ):
        """
        Perform autoregressive prediction for evaluation.

        Args:
            dynamic_seq: [B, T, dynamic_dim] - Ground truth dynamic states
            action_seq: [B, T, action_dim] - Actions
            extension_seq: Optional [B, T, extension_dim]
            contact_seq: Optional [B, T, contact_dim]
            termination_seq: Optional [B, T, termination_dim]

        Returns:
            dynamic_pred: [B, T, dynamic_dim] - Predicted dynamic states
            extension_pred: Optional [B, T, extension_dim]
            contact_pred: Optional [B, T, contact_dim]
            termination_pred: Optional [B, T, termination_dim]
            reward_pred: Optional [B, T, reward_dim]
        """
        B, T, _ = dynamic_seq.shape
        H = self.history_horizon

        # Initialize predictions with ground truth for history horizon
        dynamic_pred = dynamic_seq.clone()
        extension_pred = extension_seq.clone() if extension_seq is not None else None
        contact_pred = contact_seq.clone() if contact_seq is not None else None
        termination_pred = termination_seq.clone() if termination_seq is not None else None
        reward_pred = None

        # Autoregressive prediction for remaining steps
        x_state_batch = dynamic_seq[:, :H].clone()  # [B, H, D_d]

        for i in range(H, T):
            # Get action window
            action_window = action_seq[:, i - H + 1:i + 1]  # [B, H, D_a]

            # Predict next state and auxiliary signals
            next_dynamic, ext, contact, term, reward = self._forward_core(
                x_state_batch,  # [B, H, D_d]
                action_window,  # [B, H, D_a]
            )

            # Store predictions
            dynamic_pred[:, i] = next_dynamic
            if extension_pred is not None and ext is not None:
                extension_pred[:, i] = ext
            if contact_pred is not None and contact is not None:
                contact_pred[:, i] = torch.sigmoid(contact)  # Convert logits to probabilities
            if termination_pred is not None and term is not None:
                termination_pred[:, i] = torch.sigmoid(term)  # Convert logits to probabilities
            if reward is not None:
                if reward_pred is None:
                    reward_pred = torch.zeros(B, T, self.reward_dim, device=dynamic_seq.device)
                reward_pred[:, i] = reward

            # Update state window with predicted state (autoregressive)
            x_state_batch = torch.cat(
                [
                    x_state_batch[:, 1:].clone(),  # [B, H-1, D_d]
                    next_dynamic.unsqueeze(1),  # [B, 1, D_d]
                ],
                dim=1,
            )  # [B, H, D_d]

        return dynamic_pred, extension_pred, contact_pred, termination_pred, reward_pred


@configclass
class SystemDynamicsTransformerCfg(SystemDynamicsBaseCfg):
    """
    Configuration for Transformer-based system dynamics model.

    This mirrors SystemDynamicsMLPCfg but replaces the MLP trunk with
    a generic Transformer backbone.
    """

    class_type: type[nn.Module] = SystemDynamicsTransformer

    # Transformer backbone configuration controlling the internal representation.
    backbone_cfg: TransformerBackboneCfg = TransformerBackboneCfg(
        dim=256,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        max_seq_len=32,
        positional_encoding="learned",
        causal=False,
        use_final_layer_norm=True,
    )

