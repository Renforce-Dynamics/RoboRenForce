from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as D
from typing import Optional, Tuple, Union

from RoboRenForce import configclass
from RoboRenForce.networks.mlp import MLPCfg
from RoboRenForce.utils.template.module_base import ModuleBaseCfg
from RoboRenForce.components.actor.actor_base import ActorBase
from RoboRenForce.components.actor.state_ind_std_actor import StateIndStdActor, StateIndStdActorCfg
from dataclasses import MISSING


class BeliefEncoderActor(ActorBase):
    """
    Actor with built-in encoder for belief state extraction.
    
    Supports two forward modes:
    1. forward_with_obs(obs, slow_obs): Encode obs + slow_obs to get belief
    2. forward_with_belief(obs, belief): Use provided belief directly
    
    The encoder can be RNN-based or single-frame (MLP-based).
    After each forward, the last belief is stored in self.belief.
    """
    
    def __init__(
        self,
        cfg: "BeliefEncoderActorCfg",
        obs_dim: int,
        slow_obs_dim: int,
        belief_dim: int,
        action_dim: int,
    ):
        super().__init__(state_dim=obs_dim + belief_dim, action_dim=action_dim)
        
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.slow_obs_dim = slow_obs_dim
        self.belief_dim = belief_dim
        
        # Encoder: (obs, slow_obs) -> belief
        # Can be RNN or single-frame encoder
        self.encoder = cfg.encoder_cfg.class_type(
            cfg=cfg.encoder_cfg,
            in_feature=obs_dim + slow_obs_dim,
            out_feature=belief_dim,
        )
        
        # Base actor: (obs, belief) -> action
        self.base_actor = cfg.actor_cfg.class_type(
            cfg=cfg.actor_cfg,
            state_dim=obs_dim + belief_dim,
            action_dim=action_dim,
        )
        
        # Store last belief state
        self.belief: Optional[torch.Tensor] = None
        
        # For RNN encoders, track hidden state
        self._hidden_state = None
    
    def forward_with_obs(
        self, 
        obs: torch.Tensor, 
        slow_obs: torch.Tensor
    ) -> D.Distribution:
        """
        Forward pass using observations (obs + slow_obs).
        Encodes to belief, then passes to base actor.
        
        Args:
            obs: [B, obs_dim] observation
            slow_obs: [B, slow_obs_dim] slow observation
            
        Returns:
            Action distribution
        """
        # Concatenate obs and slow_obs
        combined_obs = torch.cat([obs, slow_obs], dim=-1)  # [B, obs_dim + slow_obs_dim]
        
        # Encode to belief
        if hasattr(self.encoder, 'forward_step') and self._hidden_state is not None:
            # RNN encoder with hidden state
            belief, self._hidden_state = self.encoder.forward_step(
                combined_obs, self._hidden_state
            )
        else:
            # Single-frame encoder or first step of RNN
            belief = self.encoder(combined_obs)
            if hasattr(self.encoder, 'get_hidden_state'):
                self._hidden_state = self.encoder.get_hidden_state()
        
        # Store belief
        self.belief = belief
        
        # Combine obs and belief for base actor
        state = torch.cat([obs, belief], dim=-1)  # [B, obs_dim + belief_dim]
        
        # Forward through base actor
        return self.base_actor(state)
    
    def forward_with_belief(
        self,
        obs: torch.Tensor,
        belief: torch.Tensor
    ) -> D.Distribution:
        """
        Forward pass using provided belief.
        
        Args:
            obs: [B, obs_dim] observation
            belief: [B, belief_dim] belief state
            
        Returns:
            Action distribution
        """
        # Store provided belief
        self.belief = belief
        
        # Combine obs and belief for base actor
        state = torch.cat([obs, belief], dim=-1)  # [B, obs_dim + belief_dim]
        
        # Forward through base actor
        return self.base_actor(state)
    
    def forward(self, state: torch.Tensor) -> D.Distribution:
        """
        Standard forward (for compatibility).
        Assumes state is already [obs, belief] concatenated.
        """
        return self.base_actor(state)
    
    # --------------------------------------------------------------------- #
    # Algorithm-facing helpers (delegate to base_actor)
    # --------------------------------------------------------------------- #
    
    def sample(
        self,
        state: torch.Tensor | None = None,
        obs: torch.Tensor | None = None,
        slow_obs: torch.Tensor | None = None,
        belief: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample action."""
        if state is not None:
            dist = self.base_actor(state)
        elif obs is not None and slow_obs is not None:
            dist = self.forward_with_obs(obs, slow_obs)
        elif obs is not None and belief is not None:
            dist = self.forward_with_belief(obs, belief)
        else:
            raise ValueError("Must provide either (state) or (obs, slow_obs) or (obs, belief)")
        
        if deterministic:
            if isinstance(dist, D.TransformedDistribution):
                return dist.base_dist.mean.tanh()
            return dist.mean
        return dist.rsample()
    
    def act(
        self,
        obs: torch.Tensor | None = None,
        slow_obs: torch.Tensor | None = None,
        belief: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample action (non-deterministic)."""
        if obs is not None and slow_obs is not None:
            dist = self.forward_with_obs(obs, slow_obs)
        elif obs is not None and belief is not None:
            dist = self.forward_with_belief(obs, belief)
        elif obs is not None:
            dist = self.base_actor(obs)
        else:
            raise ValueError("Must provide either (state) or (obs, slow_obs) or (obs, belief)")
        return dist.sample()
    
    def get_actions_log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.base_actor.get_actions_log_prob(action)
    
    @property
    def action_mean(self) -> torch.Tensor:
        return self.base_actor.action_mean
    
    @property
    def action_std(self):
        return self.base_actor.action_std
    
    def entropy(
        self,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state is not None:
            return self.base_actor.entropy(state)
        return self.base_actor.entropy()
    
    def reset(self, done: torch.Tensor | None = None):
        """Reset encoder hidden state (for RNN encoders)."""
        if hasattr(self.encoder, 'reset'):
            self.encoder.reset(done)
        self._hidden_state = None
        if hasattr(self.base_actor, 'reset'):
            self.base_actor.reset(done)
    
    @torch.no_grad()
    def act_inference(
        self,
        state: torch.Tensor | None = None,
        obs: torch.Tensor | None = None,
        slow_obs: torch.Tensor | None = None,
        belief: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic action for inference."""
        if state is not None:
            return self.base_actor.act_inference(state)
        elif obs is not None and slow_obs is not None:
            dist = self.forward_with_obs(obs, slow_obs)
        elif obs is not None and belief is not None:
            dist = self.forward_with_belief(obs, belief)
        else:
            raise ValueError("Must provide either (state) or (obs, slow_obs) or (obs, belief)")
        
        if isinstance(dist, D.TransformedDistribution):
            return dist.base_dist.mean.tanh()
        return dist.mean


@configclass
class BeliefEncoderActorCfg(ModuleBaseCfg):
    """Configuration for BeliefEncoderActor."""
    
    class_type: type[nn.Module] = BeliefEncoderActor
    
    # Encoder configuration (can be RNN or single-frame)
    encoder_cfg: ModuleBaseCfg = MISSING
    
    # Base actor configuration
    actor_cfg: StateIndStdActorCfg = StateIndStdActorCfg()
    
    def construct_from_cfg(
        self, 
        *args, 
        dim_params: dict = None, 
        **kwargs
    ):
        if dim_params is None:
            return super().construct_from_cfg(*args, **kwargs)
        
        return BeliefEncoderActor(
            cfg=self,
            obs_dim=dim_params["policy_dim"],
            slow_obs_dim=dim_params.get("slow_dim", 0),
            belief_dim=dim_params.get("belief_dim", 64),
            action_dim=dim_params["action_dim"],
        )
