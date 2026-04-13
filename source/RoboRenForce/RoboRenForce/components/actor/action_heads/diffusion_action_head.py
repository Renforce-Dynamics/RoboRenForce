"""
Diffusion Action Head (System 1)

Diffusion Transformer for action prediction using DDIM sampling.

Reference: .references/Psi0/src/psi/models/action_head.py

TODO Phase 2 (Week 2, Priority P0):
- [ ] Implement Transformer-based noise predictor
- [ ] Implement DDIM sampling
- [ ] Add noise schedule (cosine/linear)
- [ ] Implement training forward (denoising loss)
- [ ] Support action chunking (future timesteps)
- [ ] Add conditioning on fused features
"""

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass, MISSING
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


@configclass
class DiffusionActionHeadCfg(ModuleBaseCfg):
    """
    Diffusion Transformer action head configuration.

    Predicts actions via denoising diffusion process.
    """

    class_type: type["DiffusionActionHead"] = MISSING

    # Transformer architecture
    num_layers: int = 4
    num_heads: int = 8
    embed_dim: int = 256
    mlp_ratio: float = 4.0

    # Diffusion parameters
    num_diffusion_steps: int = 10  # DDIM sampling steps
    noise_schedule: str = "cosine"  # "linear" or "cosine"

    # Action prediction
    action_horizon: int = 1  # Number of future actions to predict
    action_dim: int = MISSING


class DiffusionActionHead(ModuleBase):
    """
    Diffusion-based action head using DDIM sampling.

    Training:
    - Add noise to ground-truth actions
    - Predict noise with Transformer conditioned on features
    - Minimize denoising loss

    Inference:
    - Start from Gaussian noise
    - Iteratively denoise using DDIM sampling
    - Return final denoised action

    TODO Phase 2:
    - [ ] Implement noise prediction Transformer
    - [ ] Register noise schedule (alpha, beta)
    - [ ] Implement training forward (add noise + predict)
    - [ ] Implement DDIM sampling for inference
    - [ ] Add timestep encoding
    """

    def __init__(self, cfg: DiffusionActionHeadCfg, dim_params: dict):
        super().__init__(cfg)

        input_dim = dim_params["input_dim"]
        action_dim = cfg.action_dim
        action_horizon = cfg.action_horizon

        # TODO: Build noise prediction network (Transformer)
        # - Input: noisy actions + timestep + conditioning features
        # - Output: predicted noise
        # self.noise_predictor = TransformerBackbone(...)
        raise NotImplementedError("TODO: Implement noise predictor")

        # TODO: Register noise schedule
        # self.register_noise_schedule(cfg.noise_schedule, cfg.num_diffusion_steps)
        raise NotImplementedError("TODO: Register noise schedule")

    def forward(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Predict actions.

        Args:
            features: (B, input_dim) fused features from fusion layer
            deterministic: If True, use DDIM sampling; else training mode

        Returns:
            actions: (B, action_horizon, action_dim)

        TODO:
        - If training: return structure for loss computation
        - If inference: run DDIM sampling
        """
        if self.training:
            # TODO: Training forward (used in loss computation)
            # Return predicted noise or other training outputs
            raise NotImplementedError("TODO: Implement training forward")
        else:
            # TODO: Inference forward (DDIM sampling)
            return self.sample_actions(features, num_steps=self.cfg.num_diffusion_steps)

    def train_forward(self, features: torch.Tensor, target_actions: torch.Tensor):
        """
        Training forward pass for loss computation.

        Args:
            features: (B, input_dim) conditioning features
            target_actions: (B, action_horizon, action_dim) ground-truth actions

        Returns:
            Dict with noise_pred, noise_target, etc. for loss

        TODO:
        - Sample random timestep t
        - Add noise to target_actions according to schedule
        - Predict noise with noise_predictor
        - Return predictions and targets
        """
        raise NotImplementedError("TODO: Implement train_forward")

    def sample_actions(
        self,
        features: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """
        DDIM sampling to generate actions.

        Args:
            features: (B, input_dim) conditioning features
            num_steps: Number of denoising steps

        Returns:
            actions: (B, action_horizon, action_dim) denoised actions

        TODO:
        - Start from Gaussian noise: x_T ~ N(0, I)
        - For t = T, T-1, ..., 1:
        -     Predict noise: eps = noise_predictor(x_t, t, features)
        -     Denoise: x_{t-1} = DDIM_step(x_t, eps, t)
        - Return x_0 (final denoised action)
        """
        raise NotImplementedError("TODO: Implement DDIM sampling")

    def register_noise_schedule(self, schedule_type: str, num_steps: int):
        """
        Register noise schedule (alpha, beta, alpha_bar).

        TODO:
        - Compute betas based on schedule_type
        - Compute alphas = 1 - betas
        - Compute alpha_bar (cumulative product of alphas)
        - Register as buffers
        """
        raise NotImplementedError("TODO: Register noise schedule")
