"""
Diffusion Action Head (System 1)

Diffusion Transformer for action prediction using DDIM sampling.

IO Contract:
    Inference (forward with self.training=False):
        Input:  features (B, input_dim)
        Output: actions  (B, action_horizon, action_dim)

    Training (train_forward):
        Input:  features (B, input_dim), target_actions (B, action_horizon, action_dim)
        Output: {"noise_pred": (B, H, A), "noise_target": (B, H, A), "timesteps": (B,)}

    DDIM Sampling (sample_actions):
        Input:  features (B, input_dim), num_steps int
        Output: actions  (B, action_horizon, action_dim)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg


# --------------------------------------------------------------------------- #
#  Sinusoidal timestep embedding
# --------------------------------------------------------------------------- #

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# --------------------------------------------------------------------------- #
#  Transformer block with AdaLayerNorm for timestep conditioning
# --------------------------------------------------------------------------- #

class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale_shift = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.scale_shift(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = AdaLayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x, cond)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x, cond))
        return x


# --------------------------------------------------------------------------- #
#  Noise predictor network
# --------------------------------------------------------------------------- #

class NoisePredictor(nn.Module):
    def __init__(self, action_dim, action_horizon, feature_dim, embed_dim,
                 num_layers, num_heads, mlp_ratio):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.feature_proj = nn.Linear(feature_dim, embed_dim)
        self.action_proj = nn.Linear(action_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, action_horizon + 1, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, action_dim)
        self.action_horizon = action_horizon

    def forward(self, noisy_actions, timestep, features):
        B = noisy_actions.shape[0]
        t_emb = self.time_embed(timestep)
        f_emb = self.feature_proj(features)
        cond = t_emb + f_emb

        a_emb = self.action_proj(noisy_actions)
        cls_token = f_emb.unsqueeze(1)
        tokens = torch.cat([cls_token, a_emb], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        for block in self.blocks:
            tokens = block(tokens, cond.unsqueeze(1).expand(-1, tokens.shape[1], -1))

        action_tokens = self.norm(tokens[:, 1:, :])
        return self.head(action_tokens)


# --------------------------------------------------------------------------- #
#  Diffusion Action Head
# --------------------------------------------------------------------------- #

class DiffusionActionHead(ModuleBase):
    """
    Diffusion-based action head using DDIM sampling.

    Training: add noise to GT actions, predict noise, minimize MSE.
    Inference: start from Gaussian noise, iteratively denoise via DDIM.
    """

    def __init__(self, cfg: DiffusionActionHeadCfg, dim_params: dict):
        super().__init__()
        self.cfg = cfg

        input_dim = dim_params["input_dim"]
        action_dim = cfg.action_dim
        action_horizon = cfg.action_horizon

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_train_steps = cfg.num_train_steps

        self.noise_predictor = NoisePredictor(
            action_dim=action_dim,
            action_horizon=action_horizon,
            feature_dim=input_dim,
            embed_dim=cfg.embed_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
        )

        self._register_noise_schedule(cfg.noise_schedule, cfg.num_train_steps)

    def _register_noise_schedule(self, schedule_type: str, num_steps: int):
        if schedule_type == "linear":
            betas = torch.linspace(1e-4, 0.02, num_steps)
        elif schedule_type == "cosine":
            steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = torch.clamp(betas, max=0.999).float()
        else:
            raise ValueError(f"Unknown noise schedule: {schedule_type}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def forward(self, features, deterministic=False):
        if self.training:
            raise RuntimeError("Use train_forward() during training to get loss targets.")
        return self.sample_actions(features, num_steps=self.cfg.num_diffusion_steps)

    def train_forward(self, features, target_actions):
        B = features.shape[0]
        device = features.device

        t = torch.randint(0, self.num_train_steps, (B,), device=device)
        noise = torch.randn_like(target_actions)

        sqrt_ab = self.sqrt_alpha_bar[t].view(B, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1)
        noisy_actions = sqrt_ab * target_actions + sqrt_one_minus_ab * noise

        noise_pred = self.noise_predictor(noisy_actions, t, features)

        return {"noise_pred": noise_pred, "noise_target": noise, "timesteps": t}

    @torch.no_grad()
    def sample_actions(self, features, num_steps):
        B = features.shape[0]
        device = features.device

        x = torch.randn(B, self.action_horizon, self.action_dim, device=device)

        step_ratio = self.num_train_steps // num_steps
        timesteps = torch.arange(num_steps - 1, -1, -1, device=device) * step_ratio

        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)
            noise_pred = self.noise_predictor(x, t_batch, features)

            sqrt_ab_t = self.sqrt_alpha_bar[t]
            sqrt_one_minus_ab_t = self.sqrt_one_minus_alpha_bar[t]

            x0_pred = (x - sqrt_one_minus_ab_t * noise_pred) / sqrt_ab_t

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                sqrt_ab_t_prev = self.sqrt_alpha_bar[t_prev]
                sqrt_one_minus_ab_t_prev = self.sqrt_one_minus_alpha_bar[t_prev]
                x = sqrt_ab_t_prev * x0_pred + sqrt_one_minus_ab_t_prev * noise_pred
            else:
                x = x0_pred

        return x


@configclass
class DiffusionActionHeadCfg(ModuleBaseCfg):
    """Diffusion Transformer action head configuration."""

    class_type: type[DiffusionActionHead] = DiffusionActionHead

    num_layers: int = 4
    num_heads: int = 8
    embed_dim: int = 256
    mlp_ratio: float = 4.0

    num_train_steps: int = 100
    num_diffusion_steps: int = 10
    noise_schedule: str = "cosine"

    action_horizon: int = 1
    action_dim: int = 0
