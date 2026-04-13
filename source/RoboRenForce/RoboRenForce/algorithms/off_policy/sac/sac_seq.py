import torch
from RoboRenForce import configclass
from typing import Generator, Dict, Tuple, Union
from RoboRenForce.buffer import PipeBufferTransition
from .sac import SAC, SACCfg

def flatten(x):
    return x.reshape(-1, *x.shape[2:])

class SACSeq(SAC):
    # update interface
    def update(self, generator: Generator[Dict[str, torch.Tensor], None, None]):
        self.ptr_update = 0
        critic_losses, q_means, target_q_means, actor_losses, alpha_losses, alphas, entropies = [], [], [], [], [], [], []
        for minib in generator:
            params_dict: Dict[str, torch.Tensor] = minib
            critic_loss, q_mean, target_q_mean, actor_loss, alpha_loss, alpha, entropy = self._update_at_seq(
                **params_dict,
            )
            critic_losses.append(critic_loss)
            q_means.append(q_mean)
            target_q_means.append(target_q_mean)
            if actor_loss is not None: actor_losses.append(actor_loss)
            if alpha_loss is not None: alpha_losses.append(alpha_loss)
            alphas.append(alpha)
            if entropy is not None: entropies.append(entropy)
            self.ptr_update += 1
        return {
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "actor_loss": sum(actor_losses) / len(actor_losses) if actor_losses else 0.0,
            "q_mean": sum(q_means) / len(q_means),
            "target_q_mean": sum(target_q_means) / len(target_q_means),
            "alpha_loss": sum(alpha_losses) / len(alpha_losses) if alpha_losses else 0.0,
            "alpha": sum(alphas) / len(alphas),
            "entropy": sum(entropies) / len(entropies) if entropies else 0.0,
            "mini_batch_num": self.ptr_update
        }

    def _update_at_seq(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        termination: torch.Tensor,
        timeout: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Sequence SAC update.
        Inputs are full trajectories [B, T, D].
        TD backup is constructed via time shift inside the sequence.
        """
        obs = obs.transpose(0, 1)
        critic_obs = critic_obs.transpose(0, 1)
        action = action.transpose(0, 1)
        reward = reward.transpose(0, 1)
        termination = termination.transpose(0, 1)
        timeout = timeout.transpose(0, 1)
        return self._update_at_trans(
            obs             = flatten(obs[:-1]),
            critic_obs      = flatten(critic_obs[:-1]),
            actions         = flatten(action[:-1]),
            rewards         = flatten(reward[:-1]),
            next_obs        = flatten(obs[1:]),
            next_critic_obs = flatten(critic_obs[1:]),
            termination     = flatten(termination[:-1]),
            timeout         = flatten(timeout[:-1])
        )
        
@configclass
class SACSeqCfg(SACCfg):
    class_type: type[SAC] = SACSeq