from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from RoboRenForce import configclass

from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.components.actor import StateIndStdActor, SACActor, SACActorCfg
from RoboRenForce.components.critic import VNetwork
from RoboRenForce.components.actor_critic_pack import ActorCritic
from RoboRenForce.buffer.online_rollout.rollout_storage import RolloutStorage
from RoboRenForce.utils.logging import timeit

from RoboRenForce.algorithms.algorithm_base import AlgorithmBase, AlgorithmBaseCfg
from RoboRenForce.algorithms.on_policy.ppo import PPO, PPOCfg


class SACP(PPO):
    """
    Soft Actor-Critic Compatible PPO (SACP).
    
    This algorithm inherits from PPO and maintains a pure PPO optimization regime.
    Additionally, it maintains a SAC-compatible actor network (SACActor) that learns
    to mimic the PPO actor's behavior through distillation.
    
    Key Design:
    - Primary actor: StateIndStdActor (standard PPO actor, used for training)
    - SAC-compatible actor: SACActor (learns to mimic PPO actor, for SAC fine-tuning)
    - Optimization: Pure PPO (no mixing of optimization semantics)
    - SAC actor training: Distillation loss to match PPO actor's action distribution
    
    This ensures:
    - Stable PPO training (no value loss explosion)
    - SAC-compatible policy structure for seamless fine-tuning
    - No architectural mismatch when switching to SAC
    """
    actor: StateIndStdActor  # Primary PPO actor
    sac_actor: SACActor  # SAC-compatible actor (for compatibility)
    critic: VNetwork

    def __init__(
        self,
        cfg: "SACPCfg",
        actor: "StateIndStdActor",
        sac_actor: "SACActor",
        critic: "VNetwork",
        device: str = "cpu",
    ):
        # Initialize PPO with the primary actor
        super().__init__(cfg, actor, critic, device)
        
        # Store SAC actor
        self.sac_actor = sac_actor.to(device)
        
        # Separate optimizer for SAC actor (distillation learning)
        self.sac_actor_optimizer = optim.Adam(
            self.sac_actor.parameters(),
            lr=cfg.sac_actor_lr,
        )
        
        # Distillation loss coefficient
        self.distillation_coef = cfg.distillation_coef

    def train_mode(self):
        super().train_mode()
        self.sac_actor.train()

    def test_mode(self):
        super().test_mode()
        self.sac_actor.eval()

    @timeit("update_time")
    def update(self):
        """
        Update policy and value function using standard PPO, then train SAC actor
        to mimic PPO actor through distillation.
        
        Note: Distillation is performed BEFORE PPO update to avoid storage.clear() issue.
        """
        # First, train SAC actor to mimic PPO actor (before PPO update clears storage)
        mean_distillation_loss = 0.0
        num_distillation_updates = 0
        
        generator = self.storage.mini_batch_generator(
            self.cfg.num_mini_batches, self.cfg.num_learning_epochs
        )
        
        for batch in generator:
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
            
            # Get PPO actor's current distribution
            self.actor.act(obs_batch)
            ppo_mean = self.actor.action_mean  # [B, action_dim]
            ppo_std = self.actor.action_std  # [B, action_dim] or [action_dim]
            if ppo_std.dim() == 1:
                ppo_std = ppo_std.unsqueeze(0).expand_as(ppo_mean)
            
            # Get SAC actor's distribution
            sac_mean, sac_std = self.sac_actor(obs_batch)
            
            # Distillation loss: KL divergence between PPO and SAC actor distributions
            # KL(N(ppo_mean, ppo_std) || N(sac_mean, sac_std))
            # For Gaussian: KL = 0.5 * (log(sac_std^2 / ppo_std^2) + (ppo_std^2 + (ppo_mean - sac_mean)^2) / sac_std^2 - 1)
            kl_div = torch.sum(
                0.5 * (
                    torch.log((sac_std.pow(2) / (ppo_std.pow(2) + 1e-8)) + 1e-8)
                    + (ppo_std.pow(2) + (ppo_mean - sac_mean).pow(2)) / (sac_std.pow(2) + 1e-8)
                    - 1.0
                ),
                dim=-1
            )
            distillation_loss = kl_div.mean()
            
            # Alternative: MSE loss on mean (simpler, more stable)
            if self.cfg.use_mse_distillation:
                mse_loss = nn.functional.mse_loss(sac_mean, ppo_mean)
                distillation_loss = mse_loss
            
            # Update SAC actor
            self.sac_actor_optimizer.zero_grad()
            distillation_loss.backward()
            nn.utils.clip_grad_norm_(
                self.sac_actor.parameters(),
                self.cfg.max_grad_norm,
            )
            self.sac_actor_optimizer.step()
            
            mean_distillation_loss += distillation_loss.item()
            num_distillation_updates += 1
        
        # Then, perform standard PPO update (this will clear storage)
        ppo_results = super().update()
        
        # Update results with distillation loss
        ppo_results["mean_distillation_loss"] = (
            mean_distillation_loss / num_distillation_updates if num_distillation_updates > 0 else 0.0
        )
        
        return ppo_results

    def save(self, path: str):
        """Save both PPO actor and SAC actor."""
        checkpoint = {
            "ppo_actor": self.actor.state_dict(),
            "sac_actor": self.sac_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "sac_actor_optimizer": self.sac_actor_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load both PPO actor and SAC actor."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["ppo_actor"])
        self.sac_actor.load_state_dict(checkpoint["sac_actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "sac_actor_optimizer" in checkpoint:
            self.sac_actor_optimizer.load_state_dict(checkpoint["sac_actor_optimizer"])


@configclass
class SACPCfg(PPOCfg):
    """
    Configuration for SACP (Soft Actor-Critic Compatible PPO).
    
    Inherits all PPO configuration. Additionally configures:
    - SAC actor learning rate for distillation
    - Distillation loss coefficient
    - Distillation method (KL divergence or MSE)
    - SAC actor configuration
    """
    class_type: type[SACP] = SACP
    
    # SAC actor configuration (for compatibility)
    sac_actor_cfg: "SACActorCfg" = None  # Will be set in construct_from_cfg
    
    # SAC actor distillation learning
    sac_actor_lr: float = 3e-4  # Learning rate for SAC actor (distillation)
    distillation_coef: float = 1.0  # Distillation loss coefficient (currently not used, loss is separate)
    use_mse_distillation: bool = True  # Use MSE loss instead of KL divergence (more stable)
    
    def construct_from_cfg(self, actor_critic: ActorCritic, device, *args, **kwargs):
        """
        Construct SACP with both PPO actor and SAC actor.
        
        The PPO actor comes from actor_critic.actor.
        The SAC actor is constructed from sac_actor_cfg.
        """
        # Get PPO actor and critic from actor_critic
        ppo_actor = actor_critic.actor
        critic = actor_critic.critic
        
        # Construct SAC actor if config is provided
        if self.sac_actor_cfg is not None:
            sac_actor = self.sac_actor_cfg.construct_from_cfg(
                dim_params={
                    "policy_dim": ppo_actor.state_dim,
                    "action_dim": ppo_actor.action_dim,
                }
            )
        else:
            # Default SAC actor config (matching PPO actor structure)
            default_sac_cfg = SACActorCfg(
                backbone_cfg=ppo_actor.cfg.backbone_cfg,  # Share backbone structure
                hidden_dim=ppo_actor.cfg.backbone_cfg.hidden_features[-1] if ppo_actor.cfg.backbone_cfg.hidden_features else 128,
                use_tanh=True,
                log_std_min=-5.0,
                log_std_max=2.0,
            )
            sac_actor = default_sac_cfg.construct_from_cfg(
                dim_params={
                    "policy_dim": ppo_actor.state_dim,
                    "action_dim": ppo_actor.action_dim,
                }
            )
        
        return self.class_type(
            self,
            actor=ppo_actor,
            sac_actor=sac_actor,
            critic=critic,
            device=device,
        )
