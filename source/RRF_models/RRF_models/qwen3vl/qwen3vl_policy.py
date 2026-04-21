"""
Qwen3-VL Policy Adapter — BasePolicy wrapper for Qwen3-VL VLA.

Same pattern as Qwen2VLPolicy but uses Qwen3-VL backbone.
The only difference is the VLM backbone config (Qwen3VLCfg).
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
class Qwen3VLPolicyCfg:
    """Configuration for Qwen3-VL policy adapter.

    Example:
        from RoboRenForce.networks.vlm.qwen3vl import Qwen3VLCfg
        cfg = Qwen3VLPolicyCfg(
            actor_cfg=VLAActorCfg(
                vlm_backbone_cfg=Qwen3VLCfg(model_name="Qwen/Qwen3-VL-2B-Instruct"),
                action_head_cfg=RegressionActionHeadCfg(action_dim=7),
            ),
        )
    """

    actor_cfg: VLAActorCfg = MISSING

    use_value_head: bool = False
    value_hidden_dims: tuple[int, ...] = (512, 128)
    value_input_from: str = "fusion"

    proprio_dim: int = 0


class Qwen3VLPolicy(BasePolicy):
    """Qwen3-VL policy: VLM + Fusion + ActionHead wrapped as BasePolicy.

    Obs mapping (standardized → VLAActor):
        obs["main_images"]       → obs_dict["image"]        (B,H,W,C) → (B,C,H,W)
        obs["states"]            → obs_dict["proprioception"]
        obs["task_descriptions"] → obs_dict["text"]
    """

    def __init__(self, cfg: Qwen3VLPolicyCfg):
        super().__init__()
        self.cfg = cfg

        dim_params = {"proprioception_dim": cfg.proprio_dim}
        self.actor = VLAActor(cfg.actor_cfg, dim_params)

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
        obs_dict = {}
        if "main_images" in obs:
            images = obs["main_images"]
            if isinstance(images, torch.Tensor) and images.dim() == 4:
                images = images.permute(0, 3, 1, 2).contiguous()
            obs_dict["image"] = images
        if "states" in obs:
            obs_dict["proprioception"] = obs["states"]
        if "task_descriptions" in obs:
            obs_dict["text"] = obs["task_descriptions"]
        return obs_dict

    def _get_features(self, obs: dict[str, Any]) -> torch.Tensor:
        obs_dict = self._map_obs(obs)
        with torch.set_grad_enabled(not self.cfg.actor_cfg.freeze_vlm):
            vl_features = self.actor.vlm(
                image=obs_dict["image"],
                text=obs_dict.get("text") if self.cfg.actor_cfg.use_text else None,
            )
        if self.actor.fusion is not None and "proprioception" in obs_dict:
            return self.actor.fusion(vl_features, obs_dict["proprioception"])
        return vl_features

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

    def pretrain_forward(self, obs, target_actions, **kwargs):
        obs_dict = self._map_obs(obs)
        result = self.actor(obs_dict, target_actions=target_actions)
        pred = result["pred_actions"]
        target = result["target_actions"]
        action_loss = nn.functional.l1_loss(pred, target)
        return {"loss": action_loss, "action_loss": action_loss, "pred_actions": pred}

    def ppo_forward(self, obs, actions, **kwargs):
        features = self._get_features(obs)
        obs_dict = self._map_obs(obs)
        pred_actions = self.actor.action_head(features)

        if actions is not None:
            action_diff = actions.unsqueeze(1) - pred_actions if actions.dim() < pred_actions.dim() else actions - pred_actions
            logprobs = -0.5 * (action_diff ** 2).sum(dim=-1).mean(dim=-1)
        else:
            logprobs = -0.5 * (pred_actions ** 2).sum(dim=-1).mean(dim=-1)

        result = {"logprobs": logprobs, "pred_actions": pred_actions}
        if self.value_head is not None:
            result["values"] = self.value_head(features.detach())
        return result

    def freeze_backbone(self):
        self.actor.vlm.freeze_backbone()

    def unfreeze_backbone(self):
        self.actor.vlm.unfreeze_backbone()

    def get_value(self, obs):
        if self.value_head is None:
            return None
        features = self._get_features(obs)
        return self.value_head(features.detach())
