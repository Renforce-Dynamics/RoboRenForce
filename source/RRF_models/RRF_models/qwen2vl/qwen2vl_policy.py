"""
Qwen2-VL Policy Adapter — BasePolicy wrapper for Qwen2-VL VLA.

Composes the core building blocks (VLMBackbone, FusionLayer, ActionHead)
into a complete BasePolicy that works with the RRF training loops.

The actual model code stays in RoboRenForce core (networks/vlm/, components/actor/).
This adapter only does:
1. Map standardized obs dict → VLAActor obs_dict format
2. Implement ForwardType dispatch (pretrain, ppo, etc.)
3. Optional value head for RL

Reference: RLinf rlinf/models/embodiment/qwen2vl_policy.py
"""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import Any, Optional

import torch
import torch.nn as nn

from RoboRenForce.prototype.embodied import BasePolicy, ForwardType
from RoboRenForce.utils.configclass import configclass
from RoboRenForce.components.actor.vla_actor import VLAActor, VLAActorCfg
from RRF_models.modules.value_head import ValueHead


@configclass
class Qwen2VLPolicyCfg:
    """Configuration for Qwen2-VL policy adapter.

    Example (pretrain):
        cfg = Qwen2VLPolicyCfg(
            actor_cfg=VLAActorCfg(
                vlm_backbone_cfg=Qwen2VLCfg(model_name="Qwen/Qwen2-VL-2B-Instruct"),
                action_head_cfg=RegressionActionHeadCfg(action_dim=7),
            ),
        )

    Example (PPO):
        cfg = Qwen2VLPolicyCfg(
            actor_cfg=...,
            use_value_head=True,
            value_hidden_dims=(512, 128),
        )
    """

    actor_cfg: VLAActorCfg = MISSING

    # Value head (for RL)
    use_value_head: bool = False
    value_hidden_dims: tuple[int, ...] = (512, 128)
    value_input_from: str = "fusion"  # "fusion" or "vlm"

    # Proprioception
    proprio_dim: int = 0


class Qwen2VLPolicy(BasePolicy):
    """Qwen2-VL policy: VLM + Fusion + ActionHead wrapped as BasePolicy.

    Obs mapping (standardized → VLAActor):
        obs["main_images"]       → obs_dict["image"]        (B,H,W,C) → (B,C,H,W)
        obs["states"]            → obs_dict["proprioception"]
        obs["task_descriptions"] → obs_dict["text"]
    """

    def __init__(self, cfg: Qwen2VLPolicyCfg):
        super().__init__()
        self.cfg = cfg

        # Build VLA actor (VLM + fusion + action head)
        dim_params = {"proprioception_dim": cfg.proprio_dim}
        self.actor = VLAActor(cfg.actor_cfg, dim_params)

        # Optional value head for RL
        self.value_head: Optional[ValueHead] = None
        if cfg.use_value_head:
            if cfg.value_input_from == "fusion" and self.actor.fusion is not None:
                value_input_dim = self.actor.fusion.output_dim
            else:
                value_input_dim = self.actor.vlm.output_dim
            self.value_head = ValueHead(
                input_dim=value_input_dim,
                hidden_dims=cfg.value_hidden_dims,
            )

    def _map_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Map standardized obs dict to VLAActor format."""
        obs_dict = {}

        # Images: (B, H, W, C) → (B, C, H, W)
        if "main_images" in obs:
            images = obs["main_images"]
            if isinstance(images, torch.Tensor) and images.dim() == 4:
                images = images.permute(0, 3, 1, 2).contiguous()
            obs_dict["image"] = images

        # Proprioception
        if "states" in obs:
            obs_dict["proprioception"] = obs["states"]

        # Text
        if "task_descriptions" in obs:
            obs_dict["text"] = obs["task_descriptions"]

        return obs_dict

    def _get_features(self, obs: dict[str, Any]) -> torch.Tensor:
        """Extract intermediate features (for value head)."""
        obs_dict = self._map_obs(obs)

        with torch.set_grad_enabled(not self.cfg.actor_cfg.freeze_vlm):
            vl_features = self.actor.vlm(
                image=obs_dict["image"],
                text=obs_dict.get("text") if self.cfg.actor_cfg.use_text else None,
            )

        if self.actor.fusion is not None and "proprioception" in obs_dict:
            return self.actor.fusion(vl_features, obs_dict["proprioception"])
        return vl_features

    # ---- BasePolicy interface ----

    def predict_action(self, obs: dict[str, Any], **kwargs) -> torch.Tensor:
        obs_dict = self._map_obs(obs)
        with torch.no_grad():
            return self.actor(obs_dict, deterministic=True)

    def forward(
        self,
        forward_type: ForwardType = ForwardType.PRETRAIN,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if forward_type == ForwardType.INFERENCE:
            actions = self.predict_action(kwargs["obs"])
            return {"actions": actions}
        elif forward_type == ForwardType.PRETRAIN:
            return self.pretrain_forward(**kwargs)
        elif forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.PPO:
            return self.ppo_forward(**kwargs)
        elif forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        else:
            raise ValueError(f"Unknown forward type: {forward_type}")

    def pretrain_forward(
        self,
        obs: dict[str, Any],
        target_actions: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        obs_dict = self._map_obs(obs)
        result = self.actor(obs_dict, target_actions=target_actions)

        # Compute L1 loss
        pred = result["pred_actions"]
        target = result["target_actions"]
        action_loss = nn.functional.l1_loss(pred, target)

        return {
            "loss": action_loss,
            "action_loss": action_loss,
            "pred_actions": pred,
        }

    def ppo_forward(
        self,
        obs: dict[str, Any],
        actions: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        features = self._get_features(obs)

        # Predict actions for logprob computation
        obs_dict = self._map_obs(obs)
        pred_actions = self.actor.action_head(features)

        # Simple Gaussian logprob (can be extended)
        action_diff = actions.unsqueeze(1) - pred_actions if actions.dim() < pred_actions.dim() else actions - pred_actions
        logprobs = -0.5 * (action_diff ** 2).sum(dim=-1).mean(dim=-1)

        result = {
            "logprobs": logprobs,
            "pred_actions": pred_actions,
        }

        # Value estimate
        if self.value_head is not None:
            values = self.value_head(features.detach())
            result["values"] = values

        return result

    # ---- Backbone management ----

    def freeze_backbone(self):
        self.actor.vlm.freeze_backbone()

    def unfreeze_backbone(self):
        self.actor.vlm.unfreeze_backbone()

    def get_value(self, obs: dict[str, Any]) -> Optional[torch.Tensor]:
        if self.value_head is None:
            return None
        features = self._get_features(obs)
        return self.value_head(features.detach())
