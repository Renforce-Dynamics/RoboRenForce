from __future__ import annotations

import torch
import torch.nn as nn

from RoboRenForce import configclass
from RoboRenForce.algorithms.on_policy.ppo import PPO, PPOCfg
from RoboRenForce.algorithms.smooth import (
    CAPSLossMixin, CAPSCfgMixin,
    L2C2LossMixin, L2C2CfgMixin,
    LipsLossMixin, LipsCfgMixin
)
from RoboRenForce.utils.logging import timeit


class CAPSPPO(CAPSLossMixin, PPO):
    """
    CAPS (Consistency-based Action Policy Smoothness) PPO.
    
    Extends PPO with CAPS regularization based on rsl_rl implementation:
    - Temporal smoothness: L_T = ||μ_batch - act_inference(next_obs_batch)||²
    - Spatial smoothness: L_S = ||μ_batch - act_inference(obs_batch + σ*noise)||²
    - Total CAPS loss: L_CAPS = λ_T * L_T + λ_S * L_S
    
    Where:
    - μ_batch is the current policy mean actions
    - act_inference() gets deterministic actions from the actor
    - σ is the noise scale for spatial perturbation
    - λ_T, λ_S are regularization weights
    """
    
    def __init__(
        self,
        cfg: "CAPSPPOCfg",
        actor,
        critic,
        device: str = "cpu",
    ):
        super().__init__(cfg, actor, critic, device)

    def compute_loss(self, batch):
        """Compute PPO loss with additional CAPS regularization."""
        # First compute the standard PPO losses (including KL scheduling)
        base_losses = super().compute_loss(batch)

        (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            old_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) = batch

        # Recompute policy mean for current observations
        self.actor.act(obs_batch)
        mu_batch = self.actor.action_mean

        # CAPS regularization via mixin
        caps_loss = self.compute_caps_loss(obs_batch, mu_batch)

        total_loss = base_losses["total_loss"] + caps_loss

        return {
            **base_losses,
            "total_loss": total_loss,
            "caps_loss": caps_loss,
        }

@configclass
class CAPSPPOCfg(CAPSCfgMixin, PPOCfg):
    """Configuration for CAPS PPO."""
    class_type: type[CAPSPPO] = CAPSPPO
    

    

class L2C2PPO(L2C2LossMixin, PPO):
    """
    L2C2 (Lipschitz-Constrained Policy and Critic) PPO.
    
    Extends PPO with L2C2 regularization based on rsl_rl implementation:
    - Interpolation weights: mix_weights = cont_batch * (rand - 0.5) * 2.0
    - Interpolated observations: mix_obs = obs + mix_weights * (next_obs - obs)
    - Policy smoothness: L_{s,π} = ||μ_batch - act_inference(mix_obs)||²
    - Value smoothness: L_{s,V} = ||value_batch - evaluate(mix_critic_obs)||²
    - Total L2C2 loss: L_L2C2 = λ_π * L_{s,π} + λ_V * L_{s,V}
    
    Where:
    - μ_batch is the current policy mean actions
    - act_inference() gets deterministic actions from the actor
    - λ_π, λ_V are regularization weights for policy and value function
    - cont_batch controls the interpolation range
    """
    
    def __init__(
        self,
        cfg: "L2C2PPOCfg",
        actor,
        critic,
        device: str = "cpu",
    ):
        super().__init__(cfg, actor, critic, device)

    def compute_loss(self, batch):
        """Compute PPO loss with additional L2C2 regularization."""
        # First compute the standard PPO losses (including KL scheduling)
        base_losses = super().compute_loss(batch)

        (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            old_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) = batch

        # Recompute actor / critic outputs needed for L2C2
        self.actor.act(obs_batch)
        mu_batch = self.actor.action_mean
        value_pred = self.critic(critic_obs_batch)

        l2c2_loss = self.compute_l2c2_loss(
            obs_batch=obs_batch,
            critic_obs_batch=critic_obs_batch,
            mu_batch=mu_batch,
            value_pred=value_pred,
        )

        total_loss = base_losses["total_loss"] + l2c2_loss

        return {
            **base_losses,
            "total_loss": total_loss,
            "l2c2_loss": l2c2_loss,
        }

@configclass
class L2C2PPOCfg(L2C2CfgMixin, PPOCfg):
    """Configuration for L2C2 PPO."""
    class_type: type[L2C2PPO] = L2C2PPO


class LipsPPO(LipsLossMixin, PPO):
    """
    LipsNet++ PPO with Lipschitz regularization.
    
    Extends PPO with Lipschitz constraint regularization as described in LipsNet++:
    L_Lips = λ_l ||∇_{s_t} π_θ(s_t)||
    
    The gradient norm of the policy network w.r.t. input states is penalized
    to ensure that similar input states correspond to similar output actions.
    
    Where:
    - π_θ(s_t) is the policy network output (action mean)
    - ∇_{s_t} π_θ(s_t) is the Jacobian of policy output w.r.t. input state
    - λ_l is the Lipschitz regularization weight
    """
    
    def __init__(
        self,
        cfg: "LipsPPOCfg",
        actor,
        critic,
        device: str = "cpu",
    ):
        super().__init__(cfg, actor, critic, device)
    def compute_loss(self, batch):
        """Compute PPO loss with additional Lipschitz regularization."""
        # First compute the standard PPO losses (including KL scheduling)
        base_losses = super().compute_loss(batch)

        (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            old_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) = batch

        lips_loss = torch.zeros((), device=obs_batch.device, dtype=obs_batch.dtype)

        # Lipschitz Regularization: L_Lips = λ_l ||∇_{s_t} π_θ(s_t)||
        if self.lips_lambda > 0:
            # Enable gradient computation for input observations
            obs_batch_grad = obs_batch.clone().detach().requires_grad_(True)

            # Forward pass through actor to get policy mean (do NOT sample actions here)
            _ = self.actor(obs_batch_grad)
            policy_mean = self.actor.action_mean

            # Compute gradients of policy output w.r.t. input states
            batch_size, action_dim = policy_mean.shape
            grad_norms = []

            for i in range(action_dim):
                grad_outputs = torch.zeros_like(policy_mean)
                grad_outputs[:, i] = 1.0

                grads = torch.autograd.grad(
                    outputs=policy_mean,
                    inputs=obs_batch_grad,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]

                grad_norm = torch.norm(grads, dim=-1)  # [batch_size]
                grad_norms.append(grad_norm)

            grad_norms = torch.stack(grad_norms, dim=-1)  # [batch_size, action_dim]
            mean_grad_norm = torch.mean(grad_norms, dim=-1)  # [batch_size]

            lips_loss = self.lips_lambda * torch.mean(mean_grad_norm)

        total_loss = base_losses["total_loss"] + lips_loss

        return {
            **base_losses,
            "total_loss": total_loss,
            "lips_loss": lips_loss,
        }


@configclass
class LipsPPOCfg(LipsCfgMixin, PPOCfg):
    """Configuration for LipsNet++ PPO."""
    class_type: type[LipsPPO] = LipsPPO
    
    # Lipschitz specific parameters
    lips_lambda: float = 5e-3  # Lipschitz regularization weight λ_l




class LipsL2C2PPO(L2C2LossMixin, LipsLossMixin, PPO):
    def __init__(
        self,
        cfg: "LipsL2C2PPOCfg",
        actor,
        critic,
        device: str = "cpu",
    ):
        super().__init__(cfg, actor, critic, device)
    def compute_loss(self, batch):
        """Compute PPO loss with additional L2C2 and Lipschitz regularization."""
        # First compute the standard PPO losses (including KL scheduling)
        base_losses = super().compute_loss(batch)

        (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            old_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) = batch

        # Recompute actor / critic outputs needed for L2C2
        self.actor.act(obs_batch)
        mu_batch = self.actor.action_mean
        value_pred = self.critic(critic_obs_batch)

        l2c2_loss = self.compute_l2c2_loss(
            obs_batch=obs_batch,
            critic_obs_batch=critic_obs_batch,
            mu_batch=mu_batch,
            value_pred=value_pred,
        )

        # Lipschitz Regularization via mixin
        lips_loss = self.compute_lips_loss(obs_batch)

        total_loss = base_losses["total_loss"] + l2c2_loss + lips_loss

        return {
            **base_losses,
            "total_loss": total_loss,
            "l2c2_loss": l2c2_loss,
            "lips_loss": lips_loss,
        }

@configclass
class LipsL2C2PPOCfg(L2C2CfgMixin, LipsCfgMixin, PPOCfg):
    """Configuration for L2C2LipsPPO algorithm."""
    class_type: type[LipsL2C2PPO] = LipsL2C2PPO

    
class LipsCAPSPPO(CAPSLossMixin, LipsLossMixin, PPO):
    def __init__(
        self,
        cfg: "LipsCAPSPPOCfg",
        actor,
        critic,
        device: str = "cpu",
    ):
        super().__init__(cfg, actor, critic, device)
        
        # LipsPPO specific parameters
        self.lips_lambda = cfg.lips_lambda
    def compute_loss(self, batch):
        """Compute PPO loss with additional CAPS and Lipschitz regularization."""
        # First compute the standard PPO losses (including KL scheduling)
        base_losses = super().compute_loss(batch)

        (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            old_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) = batch

        # Recompute actor outputs needed for CAPS / Lipschitz
        self.actor.act(obs_batch)
        mu_batch = self.actor.action_mean

        caps_loss = self.compute_caps_loss(obs_batch, mu_batch)

        # Lipschitz Regularization via mixin
        lips_loss = self.compute_lips_loss(obs_batch)

        total_loss = base_losses["total_loss"] + caps_loss + lips_loss

        return {
            **base_losses,
            "total_loss": total_loss,
            "caps_loss": caps_loss,
            "lips_loss": lips_loss,
        }
        
@configclass
class LipsCAPSPPOCfg(CAPSCfgMixin, LipsCfgMixin, PPOCfg):
    """Configuration for L2C2LipsPPO algorithm."""
    class_type: type[LipsCAPSPPO] = LipsCAPSPPO