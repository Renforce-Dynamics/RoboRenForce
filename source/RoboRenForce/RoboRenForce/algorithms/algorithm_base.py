import torch

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.components.actor_critic_pack import ActorCritic

class AlgorithmBase(ModuleBase):
    
    optimizer = torch.optim.Optimizer
    
    def update(self):
        ...
    
    def act(self, *args, **kwargs):
        ...
    
    def process_env_step(self, *args, **kwargs):
        pass

    def train_mode(self):
        self.train()

    def test_mode(self):
        self.eval()
        
    def clip_grad_norm(self):
        ...

@configclass
class AlgorithmBaseCfg(ModuleBaseCfg):
    class_type: type[AlgorithmBase] = None
    
    def construct_from_cfg(self, actor_critic: ActorCritic, device, *args, **kwargs):
        return self.class_type(self, actor=actor_critic.actor, critic=actor_critic.critic, device=device)
