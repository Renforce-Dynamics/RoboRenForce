"""Flow-matching Action DiT head.

Reimplemented from DiT4DiT (NVIDIA) for RoboRenForce.
State_dict key structure is identical to DiT4DiT's FlowmatchingActionHead
so that pretrained checkpoints can be loaded directly.

Checkpoint loading:
    model = FlowMatchingActionDiT(cfg)
    model.load_from_dit4dit_checkpoint("path/to/steps_N_pytorch_model.pt")
"""

from __future__ import annotations

from dataclasses import field, MISSING
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from RoboRenForce import configclass
from RoboRenForce.components.nn_models.nn_model_base import NNModelBase, NNModelBaseCfg
from .cross_attention_dit import DiT


# ---------------------------------------------------------------------------
# Sub-modules (attribute names match DiT4DiT exactly for state_dict compat)
# ---------------------------------------------------------------------------

def _swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Produces sinusoidal encoding (B, T, w) from timesteps (B, T)."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()
        B, T = timesteps.shape
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class ActionEncoder(nn.Module):
    """Encodes noisy actions + diffusion timestep into embeddings.

    Attribute names: layer1, layer2, layer3, pos_encoding — matching DiT4DiT.
    """

    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   (B, T, action_dim)
        timesteps: (B,)
        returns:   (B, T, hidden_size)
        """
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected timesteps shape (B,)")

        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = _swish(self.layer2(x))
        return self.layer3(x)


class MLP(nn.Module):
    """Simple 2-layer MLP. Attribute names: layer1, layer2."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


# ---------------------------------------------------------------------------
# DiT model size presets
# ---------------------------------------------------------------------------

DIT_CONFIGS = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@configclass
class FlowMatchingActionDiTCfg(NNModelBaseCfg):
    class_type: type[nn.Module] = None  # set after class definition

    # Model variant
    dit_variant: str = "DiT-B"

    # Action / state dimensions
    action_dim: int = MISSING
    state_dim: int = 0
    action_horizon: int = MISSING

    # DiT transformer config
    dit_cross_attention_dim: int = 2048
    dit_output_dim: int = 2560
    dit_num_layers: int = 16
    dit_dropout: float = 0.2
    dit_final_dropout: bool = True
    dit_interleave_self_attention: bool = True
    dit_norm_type: str = "ada_norm"
    dit_positional_embeddings: Optional[str] = None

    # Encoder / decoder hidden dim
    hidden_size: int = 2560

    # Position embedding
    add_pos_embed: bool = True
    max_seq_len: int = 1024

    # Flow matching noise schedule
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int = 4

    # Optimizer
    learning_rate: float = 1e-4


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FlowMatchingActionDiT(NNModelBase):
    """Flow-matching action denoising head using cross-attention DiT.

    State_dict keys match DiT4DiT's FlowmatchingActionHead exactly:
      model.*              — DiT transformer
      action_encoder.*     — ActionEncoder (layer1/2/3 + pos_encoding)
      action_decoder.*     — MLP (layer1/2)
      state_encoder.*      — MLP (layer1/2), if state_dim > 0
      position_embedding.* — nn.Embedding, if add_pos_embed
    """

    cfg: FlowMatchingActionDiTCfg

    def init_components(self):
        cfg = self.cfg
        dit_preset = DIT_CONFIGS[cfg.dit_variant]
        input_embedding_dim = dit_preset["input_embedding_dim"]

        # DiT transformer
        self.model = DiT(
            num_attention_heads=dit_preset["num_attention_heads"],
            attention_head_dim=dit_preset["attention_head_dim"],
            output_dim=cfg.dit_output_dim,
            num_layers=cfg.dit_num_layers,
            dropout=cfg.dit_dropout,
            attention_bias=True,
            activation_fn="gelu-approximate",
            upcast_attention=False,
            norm_type=cfg.dit_norm_type,
            norm_elementwise_affine=False,
            norm_eps=1e-5,
            final_dropout=cfg.dit_final_dropout,
            positional_embeddings=cfg.dit_positional_embeddings,
            interleave_self_attention=cfg.dit_interleave_self_attention,
            cross_attention_dim=cfg.dit_cross_attention_dim,
        )

        # Action encoder / decoder
        self.action_encoder = ActionEncoder(
            action_dim=cfg.action_dim,
            hidden_size=input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=cfg.dit_output_dim,
            hidden_dim=cfg.hidden_size,
            output_dim=cfg.action_dim,
        )

        # State encoder (optional)
        if cfg.state_dim > 0:
            self.state_encoder = MLP(
                input_dim=cfg.state_dim,
                hidden_dim=cfg.hidden_size,
                output_dim=input_embedding_dim,
            )
        else:
            self.state_encoder = None

        # Positional embedding
        if cfg.add_pos_embed:
            self.position_embedding = nn.Embedding(cfg.max_seq_len, input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Flow matching schedule (not a parameter)
        self.beta_dist = Beta(cfg.noise_beta_alpha, cfg.noise_beta_beta)

    def init_optimizers(self):
        super().init_optimizers()
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.learning_rate,
            betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-8,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return sample / self.cfg.noise_s

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        """Compute flow-matching loss.

        Args:
            vl_embs: (B, S, D) — conditioning embeddings (e.g. from VLM).
            actions: (B, T, action_dim) — ground-truth action trajectory.
            action_mask: (B, T, action_dim) — mask for valid action dims.
            state: (B, state_dim) — optional proprioceptive state.
            encoder_attention_mask: optional mask for cross-attention.

        Returns:
            Scalar loss.
        """
        device = vl_embs.device

        # Add noise
        noise = torch.randn_like(actions)
        t = self._sample_time(actions.shape[0], device, actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * actions + t * noise
        velocity = noise - actions

        t_discretized = (t[:, 0, 0] * self.cfg.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = (
            self.state_encoder(state).unsqueeze(1)
            if state is not None and self.state_encoder is not None
            else None
        )

        if self.cfg.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = (
            torch.cat((state_features, action_features), dim=1)
            if state_features is not None
            else action_features
        )

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
        )

        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1]:]

        loss = ((pred_actions - velocity) ** 2) * action_mask
        return loss.sum() / action_mask.sum()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise actions via Euler integration (t: 1 -> 0).

        Args:
            vl_embs: (B, S, D) conditioning embeddings.
            state: (B, state_dim) optional state.

        Returns:
            (B, action_horizon, action_dim) predicted actions.
        """
        cfg = self.cfg
        B = vl_embs.shape[0]
        device = vl_embs.device

        actions = torch.randn(
            B, cfg.action_horizon, cfg.action_dim,
            dtype=vl_embs.dtype, device=device,
        )

        num_steps = cfg.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = (
            self.state_encoder(state).unsqueeze(1)
            if state is not None and self.state_encoder is not None
            else None
        )

        for t in range(num_steps):
            t_cont = 1.0 - t / float(num_steps)
            t_disc = int(t_cont * cfg.num_timestep_buckets)
            timesteps_tensor = torch.full((B,), t_disc, device=device)

            action_features = self.action_encoder(actions, timesteps_tensor)

            if cfg.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            sa_embs = (
                torch.cat((state_features, action_features), dim=1)
                if state_features is not None
                else action_features
            )

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -cfg.action_horizon:]

            actions = actions - dt * pred_velocity

        return actions

    # ------------------------------------------------------------------
    # NNModelBase interface
    # ------------------------------------------------------------------

    def update(self, obs, action, reward, next_obs, termination, is_valid=None):
        raise NotImplementedError(
            "FlowMatchingActionDiT.update() is not used directly. "
            "Use forward() with VLM embeddings instead."
        )

    def predict_next(self, state, action):
        raise NotImplementedError("Use predict_action() instead.")

    def evaluate(self, is_valid=None, **kwargs):
        return {}

    @property
    def comp_dim(self):
        return {
            "action_dim": self.cfg.action_dim,
            "state_dim": self.cfg.state_dim,
        }

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_from_dit4dit_checkpoint(
        self,
        checkpoint_path: str,
        prefix: str = "action_model.",
        strict: bool = True,
    ):
        """Load weights from a DiT4DiT checkpoint.

        DiT4DiT checkpoints contain the full model (backbone + action_model).
        This method extracts keys with the given prefix and loads them.

        Args:
            checkpoint_path: Path to steps_N_pytorch_model.pt
            prefix: Key prefix to filter (default "action_model.")
            strict: Whether to require all keys to match.
        """
        full_state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Filter and strip prefix
        action_state_dict = {}
        for k, v in full_state_dict.items():
            if k.startswith(prefix):
                action_state_dict[k[len(prefix):]] = v

        if not action_state_dict:
            print(
                f"[WARNING] No keys found with prefix '{prefix}'. "
                "Trying to load the checkpoint as-is."
            )
            action_state_dict = full_state_dict

        # Handle diffusers ConfigMixin keys (config attribute stored by ModelMixin)
        action_state_dict = {
            k: v for k, v in action_state_dict.items()
            if not k.startswith("model.config")
        }

        missing, unexpected = self.load_state_dict(action_state_dict, strict=strict)
        if missing:
            print(f"[WARNING] Missing keys: {missing}")
        if unexpected:
            print(f"[WARNING] Unexpected keys: {unexpected}")
        print(
            f"[ActionDiT] Loaded {len(action_state_dict)} parameters "
            f"from DiT4DiT checkpoint: {checkpoint_path}"
        )


# Fix circular reference in config
FlowMatchingActionDiTCfg.class_type = FlowMatchingActionDiT
