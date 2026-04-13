from __future__ import annotations

from dataclasses import MISSING
from typing import Dict

import torch
import torch.nn as nn

from RoboRenForce import configclass
from RoboRenForce.components.actor.actor_base import ActorBase
from RoboRenForce.components.actor.state_ind_std_actor import StateIndStdActor, StateIndStdActorCfg
from RoboRenForce.components.encoder.vec_state_encoder import VecStateEncoder, VecStateEncoderCfg
from RoboRenForce.utils.template.module_base import ModuleBaseCfg


class EncoderStateActor(ActorBase):
    """
    Actor that first encodes raw observations with a VecStateEncoder and then
    applies a standard Gaussian policy (StateIndStdActor) on the encoded state.

    This mirrors the idea of EncoderActorCritic in instinct_rl, but only for the actor.
    """

    def __init__(
        self,
        cfg: "EncoderStateActorCfg",
        obs_dim: int,
        action_dim: int,
    ):
        super().__init__(state_dim=obs_dim, action_dim=action_dim)

        self.cfg = cfg
        self.obs_dim = obs_dim
        self.encoded_dim = cfg.encoded_dim

        # Encoder: obs_dim -> encoded_dim
        self.encoder: VecStateEncoder = cfg.encoder_cfg.class_type(
            cfg=cfg.encoder_cfg,
            in_feature=obs_dim,
            out_feature=cfg.encoded_dim,
        )

        # Underlying Gaussian actor on [state, encoded_state]
        # Input dimension: obs_dim (direct state) + encoded_dim (encoded state)
        self.base_actor: StateIndStdActor = cfg.actor_cfg.class_type(
            cfg=cfg.actor_cfg,
            state_dim=obs_dim + cfg.encoded_dim,
            action_dim=action_dim,
        )

    @property
    def action_mean(self) -> torch.Tensor:
        return self.base_actor.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.base_actor.action_std

    def _combine_states(
        self,
        state: torch.Tensor | None = None,
        encode_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Combine direct state and encoded state into a single tensor.
        
        Args:
            state: Direct state (no encoding needed)
            encode_state: State that needs to be encoded
            
        Returns:
            Combined state tensor of shape [..., obs_dim + encoded_dim]
        """
        parts = []
        
        # Add direct state if provided, otherwise use zeros
        if state is not None:
            parts.append(state)
        else:
            # Get batch shape from encode_state if available, otherwise create zeros
            if encode_state is not None:
                batch_shape = encode_state.shape[:-1]
                device = encode_state.device
                dtype = encode_state.dtype
            else:
                raise ValueError("At least one of 'state' or 'encode_state' must be provided")
            parts.append(torch.zeros(*batch_shape, self.obs_dim, device=device, dtype=dtype))
        
        # Encode state if provided, otherwise use zeros
        if encode_state is not None:
            encoded = self.encoder(encode_state)
            parts.append(encoded)
        else:
            # Get batch shape from state
            batch_shape = state.shape[:-1]
            device = state.device
            dtype = state.dtype
            parts.append(torch.zeros(*batch_shape, self.encoded_dim, device=device, dtype=dtype))
        
        # Concatenate: [state, encoded_state]
        return torch.cat(parts, dim=-1)

    def forward(self, state: torch.Tensor | None = None, encode_state: torch.Tensor | None = None):
        combined_state = self._combine_states(state, encode_state)
        return self.base_actor(combined_state)

    def act(self, state: torch.Tensor | None = None, encode_state: torch.Tensor | None = None) -> torch.Tensor:
        combined_state = self._combine_states(state, encode_state)
        return self.base_actor.act(combined_state)

    def sample(self, state: torch.Tensor | None = None, encode_state: torch.Tensor | None = None, deterministic: bool = False) -> torch.Tensor:
        """
        Sample action from the policy.
        
        Args:
            state: State that can be used directly (no encoding needed)
            encode_state: State that needs to be encoded (will be passed through encoder)
            deterministic: Whether to sample deterministically
            
        Returns:
            Sampled action tensor
            
        Note:
            The final input to base_actor is [state, encoded_state] with shape [..., obs_dim + encoded_dim].
            If one is None, zeros are used to pad the missing part.
        """
        combined_state = self._combine_states(state, encode_state)
        return self.base_actor.sample(combined_state, deterministic=deterministic)

    def get_actions_log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.base_actor.get_actions_log_prob(action)

    def entropy(self, state: torch.Tensor | None = None, encode_state: torch.Tensor | None = None) -> torch.Tensor:
        if state is not None or encode_state is not None:
            combined_state = self._combine_states(state, encode_state)
            return self.base_actor.entropy(combined_state)
        return self.base_actor.entropy()

    def reset(self, *args, **kwargs):
        self.base_actor.reset(*args, **kwargs)

    @torch.no_grad()
    def act_inference(self, state: torch.Tensor | None = None, encode_state: torch.Tensor | None = None) -> torch.Tensor:
        combined_state = self._combine_states(state, encode_state)
        return self.base_actor.act_inference(combined_state)


@configclass
class EncoderStateActorCfg(ModuleBaseCfg):
    """Configuration for EncoderStateActor."""

    class_type: type[nn.Module] = EncoderStateActor

    # Dimension of encoded state passed to the base actor
    encoded_dim: int = 256

    # Sub-configs
    encoder_cfg: VecStateEncoderCfg = MISSING
    actor_cfg: StateIndStdActorCfg = StateIndStdActorCfg()

    def construct_from_cfg(self, *args, dim_params: Dict = None, **kwargs):
        if dim_params is None:
            return super().construct_from_cfg(*args, **kwargs)
        return EncoderStateActor(
            cfg=self,
            obs_dim=dim_params["policy_dim"],
            action_dim=dim_params["action_dim"],
        )

