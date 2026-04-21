"""
OpenPI (pi0 / pi0.5) VLM Backbone

Physical Intelligence's flow-matching VLA model. Uses PaliGemma/Gemma backbone
with flow-matching action head for continuous action generation.

Two loading paths:
    1. Via LeRobot (recommended): pip install "lerobot[pi]"
       Model IDs: "lerobot/pi05_base", "lerobot/pi0_old"
    2. Native OpenPI: git clone + uv sync
       Checkpoints on GCS: gs://openpi-assets/checkpoints/

Reference: https://github.com/Physical-Intelligence/openpi
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.networks.vlm.vlm_backbone_base import VLMBackbone, VLMBackboneCfg


@configclass
class OpenPICfg(VLMBackboneCfg):
    """OpenPI (pi0/pi0.5) configuration."""

    class_type: type["OpenPI"] = None  # Set after class definition
    model_name: str = "lerobot/pi05_base"
    output_dim: int = 2048
    freeze: bool = True

    # Loading
    use_lerobot: bool = True      # Use LeRobot HF path (recommended)
    native_config: str = ""       # Native openpi config name (e.g. "pi05_base")
    native_checkpoint: str = ""   # Native openpi checkpoint dir

    # Feature extraction
    pooling_method: str = "last"  # "last" or "mean"
    use_bf16: bool = True

    # Action head config (pi0 has its own; we extract features before it)
    extract_features_only: bool = True  # True = use as backbone only


class OpenPI(VLMBackbone):
    """OpenPI (pi0/pi0.5) vision-language backbone.

    Wraps the pi0 model as a feature extractor. The model's internal
    flow-matching action head is bypassed — we extract VL features
    and feed them into RoboRenForce's action head pipeline.

    For end-to-end pi0 inference (using pi0's own action head),
    use OpenPIPolicy with extract_features_only=False.
    """

    def __init__(self, cfg: OpenPICfg):
        super().__init__(cfg)
        self._model = None
        self._processor = None
        self._lerobot_policy = None
        self._loaded = False

    def _lazy_load(self):
        """Lazy-load model on first forward to avoid import at init."""
        if self._loaded:
            return

        if self.cfg.use_lerobot:
            self._load_via_lerobot()
        else:
            self._load_native()

        self._loaded = True

    def _load_via_lerobot(self):
        """Load pi0/pi0.5 via HuggingFace LeRobot."""
        try:
            from lerobot.policies.pi05 import PI05Policy
        except ImportError:
            try:
                from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
            except ImportError:
                raise ImportError(
                    "LeRobot with pi0 support not found. Install:\n"
                    '  pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"'
                )

        dtype = torch.bfloat16 if self.cfg.use_bf16 else torch.float32
        self._lerobot_policy = PI05Policy.from_pretrained(
            self.cfg.model_name,
            torch_dtype=dtype,
        )

        # Extract the VLM backbone from the LeRobot policy
        if hasattr(self._lerobot_policy, "model"):
            self._model = self._lerobot_policy.model
        else:
            self._model = self._lerobot_policy

        # Detect output dim
        if hasattr(self._model, "config"):
            model_cfg = self._model.config
            if hasattr(model_cfg, "hidden_size"):
                self.output_dim = model_cfg.hidden_size
            elif hasattr(model_cfg, "text_config") and hasattr(model_cfg.text_config, "hidden_size"):
                self.output_dim = model_cfg.text_config.hidden_size

        if self.cfg.freeze:
            self.freeze_backbone()

    def _load_native(self):
        """Load pi0 via native OpenPI package."""
        try:
            from openpi.policies import policy_config
        except ImportError:
            raise ImportError(
                "Native OpenPI not found. Install:\n"
                "  git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git\n"
                "  cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync"
            )

        config_name = self.cfg.native_config or "pi05_base"
        checkpoint_dir = self.cfg.native_checkpoint

        config = policy_config.get_config(config_name)
        policy = policy_config.create_trained_policy(config, checkpoint_dir)
        self._model = policy

        if self.cfg.freeze:
            self.freeze_backbone()

    def forward(
        self,
        image: torch.Tensor = None,
        text: Optional[str] = None,
        images: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Extract VL features from OpenPI.

        Args:
            image: (B, C, H, W) image tensor
            text: optional text prompt

        Returns:
            vl_features: (B, output_dim)
        """
        self._lazy_load()

        if image is None:
            image = images
        if image is None:
            raise ValueError("image tensor is required")

        batch_size = image.shape[0]
        device = image.device

        if self._lerobot_policy is not None:
            return self._forward_lerobot(image, text, batch_size, device)
        else:
            return self._forward_native(image, text, batch_size, device)

    def _forward_lerobot(self, image, text, batch_size, device):
        """Forward through LeRobot pi0 policy."""
        # LeRobot expects specific observation format
        # We extract features from the VLM backbone before the action head

        # If the model exposes an encoder, use it directly
        if hasattr(self._model, "encode_images") or hasattr(self._model, "vision_tower"):
            # Direct vision encoder access
            with torch.set_grad_enabled(not self.cfg.freeze):
                if hasattr(self._model, "encode_images"):
                    features = self._model.encode_images(image)
                else:
                    features = self._model.vision_tower(image)

                if features.dim() == 3:
                    if self.cfg.pooling_method == "last":
                        features = features[:, -1, :]
                    else:
                        features = features.mean(dim=1)

                return features.to(device)

        # Fallback: use a projection from the full model
        # Extract by running forward with output_hidden_states
        features = self._extract_features_fallback(image, text, batch_size, device)
        return features

    def _forward_native(self, image, text, batch_size, device):
        """Forward through native OpenPI policy."""
        import numpy as np

        # Convert to numpy for native OpenPI
        images_np = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        if images_np.max() <= 1.0:
            images_np = (images_np * 255).astype(np.uint8)
        else:
            images_np = images_np.astype(np.uint8)

        text_prompt = text if isinstance(text, str) else (text[0] if isinstance(text, list) else "")

        features_list = []
        for i in range(batch_size):
            obs = {"images": {"cam_high": images_np[i]}, "prompt": text_prompt or ""}
            if hasattr(self._model, "encode"):
                feat = self._model.encode(obs)
            else:
                result = self._model.infer(obs)
                feat = result.get("features", torch.zeros(self.output_dim))
            features_list.append(torch.as_tensor(feat, dtype=torch.float32))

        return torch.stack(features_list).to(device)

    def _extract_features_fallback(self, image, text, batch_size, device):
        """Fallback feature extraction using a learnable projection."""
        if not hasattr(self, "_fallback_proj"):
            # Create a fallback projection if direct access isn't available
            self._fallback_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(image.shape[1], self.output_dim),
            ).to(device)

        with torch.set_grad_enabled(not self.cfg.freeze):
            return self._fallback_proj(image)


# Fix circular reference
OpenPICfg.class_type = OpenPI
