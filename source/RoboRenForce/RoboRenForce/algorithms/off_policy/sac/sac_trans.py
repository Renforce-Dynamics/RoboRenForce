import torch
from RoboRenForce import configclass
from typing import Generator, Dict, Tuple, Union
from RoboRenForce.buffer import PipeBufferTransition
from .sac import SAC, SACCfg

class SACTrans(SAC):
    
    def update(self, generator: Generator[Union[PipeBufferTransition.TransitionData, dict], None, None]):
        self.ptr_update = 0
        critic_losses, q_means, target_q_means, actor_losses, alpha_losses, alphas, entropies = [], [], [], [], [], [], []
        for minib in generator:
            if isinstance(minib, dict): pass
            elif isinstance(minib, PipeBufferTransition.TransitionData): minib = minib.to_dict()
            else: raise NotImplementedError("SAC get bad replay buffer.")
            critic_loss, q_mean, target_q_mean, actor_loss, alpha_loss, alpha, entropy = self._update_at_trans(**minib)
            critic_losses.append(critic_loss); q_means.append(q_mean); target_q_means.append(target_q_mean)
            if actor_loss is not None   : actor_losses.append(actor_loss)
            if alpha_loss is not None   : alpha_losses.append(alpha_loss)
            if entropy is not None      : entropies.append(entropy)
            alphas.append(alpha)
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
    
@configclass
class SACTransCfg(SACCfg):
    class_type: type[SAC] = SACTrans