"""
NVIDIA GR00T N1.7 Policy Adapter

Wraps GR00T's VLM backbone + RoboRenForce action head as a BasePolicy.

Architecture (N1.7):
    Vision: SigLip2 ViT
    Language: Cosmos-Reason2-2B (Qwen3-VL based)
    Action: Flow-matching DiT with AdaLN

Two modes:
    1. Feature extraction (default): GR00T VLM → RoboRenForce fusion → action head
    2. End-to-end: GR00T's own DiT action decoder
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.prototype.embodied import BasePolicy, ForwardType
from RoboRenForce.components.actor.vla_actor import VLAActor, VLAActorCfg
from RoboRenForce.networks.vlm.gr00t import GR00TCfg
from RRF_models.modules.value_head import ValueHead


@configclass
class GR00TPolicyCfg:
    """GR00T policy configuration."""

    actor_cfg: VLAActorCfg = None

    # Value head (for RL)
    use_value_head: bool = False
    value_hidden_dims: tuple[int, ...] = (512, 128)
    value_input_from: str = "fusion"

    # Proprioception
    proprio_dim: int = 0

    # Embodiment
    embodiment_tag: str = "new_embodiment"


class GR00TPolicy(BasePolicy):
    """NVIDIA GR00T N1.7 wrapped as BasePolicy.

    Standard mode: GR00T VLM backbone → fusion → RoboRenForce action head.
    """

    def __init__(self, cfg: GR00TPolicyCfg):
        super().__init__()
        self.cfg = cfg

        # Build VLA actor
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
        """Map EmbodiedEnv observations to VLAActor format."""
        obs_dict = {}

        if "main_images" in obs:
            images = obs["main_images"]
            if isinstance(images, torch.Tensor) and images.dim() == 4:
                if images.shape[-1] == 3:
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
            return {"actions": self.predict_action(kwargs["obs"])}
        elif forward_type == ForwardType.PRETRAIN:
            return self._pretrain_forward(**kwargs)
        elif forward_type == ForwardType.PPO:
            return self._ppo_forward(**kwargs)
        elif forward_type == ForwardType.SFT:
            return self._pretrain_forward(**kwargs)
        elif forward_type == ForwardType.SAC:
            return self._ppo_forward(**kwargs)
        else:
            raise ValueError(f"Unsupported forward type: {forward_type}")

    def _pretrain_forward(self, obs, target_actions, **kwargs):
        obs_dict = self._map_obs(obs)
        result = self.actor(obs_dict, target_actions=target_actions)

        pred = result["pred_actions"]
        target = result["target_actions"]
        action_loss = nn.functional.l1_loss(pred, target)

        return {"loss": action_loss, "action_loss": action_loss, "pred_actions": pred}

    def _ppo_forward(self, obs, actions=None, **kwargs):
        features = self._get_features(obs)
        pred_actions = self.actor.action_head(features)

        if actions is not None:
            diff = (actions.unsqueeze(1) - pred_actions
                    if actions.dim() < pred_actions.dim()
                    else actions - pred_actions)
            logprobs = -0.5 * (diff ** 2).sum(dim=-1).mean(dim=-1)
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

    def get_value(self, obs: dict[str, Any]) -> Optional[torch.Tensor]:
        if self.value_head is None:
            return None
        features = self._get_features(obs)
        return self.value_head(features.detach())
