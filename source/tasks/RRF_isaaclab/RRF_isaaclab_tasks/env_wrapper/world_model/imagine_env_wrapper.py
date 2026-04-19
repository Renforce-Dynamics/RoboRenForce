import torch
from ..dynamic_env_wrapper import RFDynamicEnvWrapper
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from RoboRenForce.components.world_models.system_dynamics.system_dynamics_mlp import SystemDynamicsMLP

class RFImagineEnvWrapper(RFDynamicEnvWrapper):
    """Wrapper for imagination rollouts using a learned system dynamics model.
    
    This wrapper extends RFDynamicEnvWrapper to support:
    - Extracting system observations (dynamic state, action, etc.) from the environment
    - Converting dynamic/action history into policy observations for imagination
    - Stepping the imagination environment using the system dynamics model
    """
    
    system_dynamic_model: Optional["SystemDynamicsMLP"] = None
    _last_action: Optional[torch.Tensor] = None
    
    def __init__(self, env, clip_actions=None):
        super().__init__(env, clip_actions)
        self._last_action = None
        dynamic_dim = self.dim_params["dynamic_dim"]
        policy_dim = self.dim_params["policy_dim"]
        self.dynamic_activated = dynamic_dim != policy_dim
    
    def set_system_dynamics(self, dynamic_model: "SystemDynamicsMLP"):
        """Set the system dynamics model for imagination rollouts.
        
        Args:
            dynamic_model: The learned system dynamics model (SystemDynamicsMLP)
        """
        self.system_dynamic_model = dynamic_model
        if dynamic_model is not None:
            self.system_dynamic_model.eval()
    
    def get_system_observation(self, obs_dict: dict=None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract system observations from the current environment dynamic state.
        
        Policy observation structure: [Commands, Dynamics(with noise), Last Action]
        We extract the Dynamics part for system dynamics prediction.
        
        Returns:
            system_dynamic: [num_envs, dynamic_dim] - Dynamic state for system dynamics (extracted from policy obs)
            system_action: [num_envs, action_dim] - Last action taken
            system_extension: Optional[torch.Tensor] - Extension signals (if available)
            system_contact: Optional[torch.Tensor] - Contact signals (if available)
            system_termination: Optional[torch.Tensor] - Termination signals (if available)
        """
        # Get current policy observation
        obs_dict = obs_dict if obs_dict is not None else self.obs_dict

        system_dynamic = obs_dict["dynamic"] if self.dynamic_activated else obs_dict["policy"]
        system_action = self._last_action

        # Not Used Yet
        system_extension    = None
        system_contact      = None
        system_termination  = None
        return system_dynamic, system_action, system_extension, system_contact, system_termination
    
    def get_imagination_observation(
        self,
        dynamic_history: torch.Tensor,
        action_history: torch.Tensor,
        command_history: torch.Tensor = None
    ) -> torch.Tensor:
        """Convert dynamic/action history into policy observations for imagination.
        
        Policy observation structure: [Commands, Dynamics(with noise), Last Action]
        We construct this from:
        - Commands: from command_history or current commands
        - Dynamics: from dynamic_history (last dynamic state)
        - Last Action: from action_history (last action)
        
        Args:
            dynamic_history: [num_envs, history_horizon, dynamic_dim] - History of system dynamic states (Dynamics part), once the dynamic is same shape of the policy, we do nothing.
            action_history: [num_envs, history_horizon, action_dim] - History of actions
            command_history: Optional[torch.Tensor] - History of commands [num_envs, history_horizon, command_dim]
        
        Returns:
            imagination_obs: [num_envs, policy_dim] - Policy observations for imagination
        """
        # Calculate dimensions
        command_dim = self.commad_shape
        dynamic_dim = self.dim_params["dynamic_dim"]
        action_dim = self.dim_params["action_dim"]
        policy_dim = self.dim_params["policy_dim"]
        
        if dynamic_dim == policy_dim: return dynamic_history[:, -1]
        num_imagine_envs = dynamic_history.shape[0]
        current_dynamics = dynamic_history[:, -1]  # [num_envs, dynamic_dim]
        last_action = action_history[:, -1]  # [num_envs, action_dim]
        if command_history is not None:
            current_commands = command_history[:, -1]  # [num_envs, command_dim]
        else:
            current_commands_raw = self.get_commands()
            if current_commands_raw.shape[0] < num_imagine_envs:
                repeat_times = (num_imagine_envs + current_commands_raw.shape[0] - 1) // current_commands_raw.shape[0]
                current_commands_tile = current_commands_raw.repeat(repeat_times, 1)  # repeat in num_envs dimension
                current_commands = current_commands_tile[:num_imagine_envs, ...]
            else:
                current_commands = current_commands_raw[:num_imagine_envs, ...]
            
        imagination_obs = torch.cat([current_commands, current_dynamics, last_action], dim=-1)
        assert imagination_obs.shape[-1] == policy_dim, (
            f"Constructed observation dimension {imagination_obs.shape[-1]} "
            f"does not match policy_dim {policy_dim}. "
            f"command_dim={command_dim}, dynamic_dim={dynamic_dim}, action_dim={action_dim}"
        )
        return imagination_obs
    
    def imagination_step(
        self,
        actions: torch.Tensor,
        dynamic_history: torch.Tensor,
        action_history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        """Step the imagination environment using the system dynamics model.
        
        Args:
            actions: [num_envs, action_dim] - Actions to take in imagination
            dynamic_history: [num_envs, history_horizon, dynamic_dim] - Current dynamic history (Dynamics part)
            action_history: [num_envs, history_horizon, action_dim] - Current action history
        
        Returns:
            imagination_obs_next: [num_envs, policy_dim] - Next policy observations
            imagination_rewards: [num_envs, 1] - Predicted rewards
            imagination_dones: [num_envs, 1] - Predicted termination signals
            imagination_extras: dict - Additional information (observations dict)
            updated_dynamic_history: [num_envs, history_horizon, dynamic_dim] - Updated dynamic history
            updated_action_history: [num_envs, history_horizon, action_dim] - Updated action history
        """
        assert self.system_dynamic_model is not None, "System dynamics model not set. Call set_system_dynamics() first."

        current_dynamic = dynamic_history[:, -1:]  # [num_envs, 1, dynamic_dim]
        current_action = actions.unsqueeze(1)  # [num_envs, 1, action_dim]
        dynamic_seq = torch.cat([dynamic_history, current_dynamic], dim=1)  # [num_envs, history_horizon+1, dynamic_dim]
        action_seq = torch.cat([action_history, current_action], dim=1)  # [num_envs, history_horizon+1, action_dim]
        
        # Predict next Dynamics state and reward using system dynamics model
        with torch.no_grad():
            next_dynamics_pred, extension_pred, contact_pred, termination_pred, reward_pred = self.system_dynamic_model(
                dynamic_seq, action_seq
            )
        
        # Update dynamic and action histories
        # Keep only the last history_horizon steps
        updated_dynamic_history = torch.cat([dynamic_history[:, 1:], next_dynamics_pred.unsqueeze(1)], dim=1)
        updated_action_history = torch.cat([action_history[:, 1:], current_action], dim=1)
        
        # Convert next Dynamics state to policy observation
        # Note: We need commands for the next observation, but in imagination we might keep the same commands
        # or sample new ones. For now, we'll use the current commands (they don't change in imagination)
        imagination_obs_next = self.get_imagination_observation(
            updated_dynamic_history, updated_action_history
        )
        
        # Use predicted reward from model if available, otherwise compute from dynamics
        if reward_pred is not None:
            imagination_rewards = reward_pred
            if imagination_rewards.dim() == 1: imagination_rewards = imagination_rewards.unsqueeze(-1)
        else:
            parsed_dynamics = self._parse_imagination_dynamics(next_dynamics_pred)
            parsed_extensions = self._parse_extensions(extension_pred) if extension_pred is not None else None
            parsed_contacts = self._parse_contacts(contact_pred) if contact_pred is not None else None
            self._compute_imagination_reward_terms(parsed_dynamics, actions, parsed_extensions, parsed_contacts)
            imagination_rewards = self._post_imagination_reward_step()
            if imagination_rewards.dim() == 1: imagination_rewards = imagination_rewards.unsqueeze(-1)
        
        if termination_pred.dim() > 1 and termination_pred.shape[-1] > 1:
            termination_prob = torch.sigmoid(termination_pred).mean(dim=-1, keepdim=True)
        else:
            termination_prob = torch.sigmoid(termination_pred)
        imagination_dones = (termination_prob > 0.5).to(dtype=torch.long)
        
        imagination_extras = {
            "observations": {
                "policy": imagination_obs_next,
                "critic": imagination_obs_next,  # Assume same as policy for now TODO Ensure the critic level.
            },
            "termination": imagination_dones.squeeze(-1) if imagination_dones.shape[-1] == 1 else imagination_dones,
        }
        
        return (
            imagination_obs_next,
            imagination_rewards,
            imagination_dones,
            imagination_extras,
            updated_dynamic_history,
            updated_action_history,
        )
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Override step to track last action for system observation extraction."""
        self._last_action = actions.clone()
        obs, reward, done, extras = super().step(actions)
        self.obs_dict = extras["observations"]
        return obs, reward, done, extras
    
    def _parse_imagination_dynamics(self, imagination_dynamics: torch.Tensor) -> dict:
        """Parse predicted dynamic states into named components.
        
        This is a placeholder that should be overridden by subclasses.
        The default implementation assumes the dynamic state is already in the correct format.
        
        Args:
            imagination_dynamics: [num_envs, dynamic_dim] - Predicted dynamics state
        
        Returns:
            dict: Parsed dynamic components (e.g., {"base_lin_vel": ..., "joint_pos": ...})
        """
        return {"dynamics": imagination_dynamics}
    
    def _parse_extensions(self, extensions: torch.Tensor) -> Optional[dict]:
        """Parse extension signals.
        
        Args:
            extensions: [num_envs, extension_dim] or None
        
        Returns:
            dict or None: Parsed extension components
        """
        if extensions is None: return None
        return {"extensions": extensions}
    
    def _parse_contacts(self, contacts: torch.Tensor) -> Optional[dict]:
        """Parse contact signals.
        
        Args:
            contacts: [num_envs, contact_dim] or None
        
        Returns:
            dict or None: Parsed contact components (e.g., {"foot_contact": ..., "thigh_contact": ...})
        """
        if contacts is None: return None
        contacts_binary = torch.sigmoid(contacts).round()
        return {"contacts": contacts_binary}
    
    def _compute_imagination_reward_terms(
        self,
        parsed_dynamics: dict,
        actions: torch.Tensor,
        parsed_extensions: Optional[dict] = None,
        parsed_contacts: Optional[dict] = None,
    ):
        """Compute reward terms from predicted dynamic states.
        
        This method should be overridden by subclasses to implement environment-specific
        reward computation. The default implementation returns zero rewards.
        
        Args:
            parsed_dynamics: dict - Parsed dynamic components from _parse_imagination_states
            actions: [num_envs, action_dim] - Actions taken
            parsed_extensions: Optional dict - Parsed extension signals
            parsed_contacts: Optional dict - Parsed contact signals
        
        Note:
            Subclasses should store reward terms in self.imagination_reward_per_step
            for use in _post_imagination_reward_step()
        """
        # Default: zero rewards (subclasses should override)
        # TODO, enable more rewards setting.
        num_envs = actions.shape[0]
        self.imagination_reward_per_step = {
            "reward": torch.zeros(num_envs, device=actions.device)
        }
        raise NotImplementedError("Rule based reward calculation yet not supported.")
    
    def _post_imagination_reward_step(self) -> torch.Tensor:
        """Combine reward terms into final reward.
        
        This method combines the reward terms computed in _compute_imagination_reward_terms
        using the environment's reward manager weights (if available).
        
        Returns:
            rewards: [num_envs, 1] - Final reward values
        """
        # Sum all reward terms
        reward_values = list(self.imagination_reward_per_step.values())
        if reward_values and isinstance(reward_values[0], torch.Tensor):
            total_reward = sum(reward_values)
            # Ensure correct shape [num_envs, 1]
            if total_reward.dim() == 1:
                return total_reward.unsqueeze(-1)
            elif total_reward.dim() == 0:
                # Scalar, need to know num_envs
                num_envs = self.num_envs if hasattr(self, "num_envs") else 1
                return total_reward.unsqueeze(0).unsqueeze(-1).expand(num_envs, 1)
            else:
                return total_reward
        
    """
    We assume that the policy observation is contructed as [Commands, Dynamics(with noise), Last Action]
    """
    
    @property
    def dim_params(self):
        dim_params = {
            "policy_dim": self.observation_space["policy"].shape[-1],
            "critic_dim": self.observation_space.get("critic", self.observation_space["policy"]).shape[-1],
            "dynamic_dim": self.observation_space.get("dynamic", self.observation_space["policy"]).shape[-1],
            "action_dim": self.action_space.shape[-1],
            "rewards_dim": self.rewards_shape,
            "termination_dim": 1,
        }
        return dim_params