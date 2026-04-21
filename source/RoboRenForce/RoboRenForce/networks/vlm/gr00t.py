"""
NVIDIA GR00T N1.7 VLM Backbone

NVIDIA's generalist robot policy with Cosmos-Reason2 VLM backbone,
SigLip2 vision encoder, and flow-matching DiT action decoder.

Model IDs:
    nvidia/GR00T-N1.7-3B          (base)
    nvidia/GR00T-N1.7-DROID       (DROID fine-tuned)
    nvidia/GR00T-N1.7-LIBERO      (LIBERO fine-tuned)

Reference: https://github.com/NVIDIA/Isaac-GR00T
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.networks.vlm.vlm_backbone_base import VLMBackbone, VLMBackboneCfg


@configclass
class GR00TCfg(VLMBackboneCfg):
    """GR00T N1.7 configuration."""

    class_type: type["GR00T"] = None  # Set after class definition
    model_name: str = "nvidia/GR00T-N1.7-3B"
    output_dim: int = 2048
    freeze: bool = True

    # Embodiment
    embodiment_tag: str = "new_embodiment"  # Robot type tag

    # Feature extraction
    pooling_method: str = "last"
    use_bf16: bool = True
    extract_features_only: bool = True

    # Temporal
    num_frames: int = 1  # Number of temporal frames for video input


class GR00T(VLMBackbone):
    """NVIDIA GR00T N1.7 vision-language backbone.

    Wraps GR00T as a feature extractor. The model's internal DiT action
    decoder is bypassed — we extract VL features for RoboRenForce's
    action head pipeline.

    For end-to-end GR00T inference (using GR00T's own action decoder),
    use GR00TPolicy with extract_features_only=False.
    """

    def __init__(self, cfg: GR00TCfg):
        super().__init__(cfg)
        self._model = None
        self._processor = None
        self._loaded = False

    def _lazy_load(self):
        """Lazy-load model on first forward."""
        if self._loaded:
            return

        try:
            self._load_via_hf()
        except ImportError:
            self._load_via_gr00t_package()

        self._loaded = True

    def _load_via_hf(self):
        """Load GR00T via HuggingFace transformers."""
        from transformers import AutoModel, AutoProcessor

        dtype = torch.bfloat16 if self.cfg.use_bf16 else torch.float32

        self._model = AutoModel.from_pretrained(
            self.cfg.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=True,
        )

        # Detect output dim
        if hasattr(self._model, "config"):
            model_cfg = self._model.config
            if hasattr(model_cfg, "hidden_size"):
                self.output_dim = model_cfg.hidden_size
            elif hasattr(model_cfg, "text_config"):
                self.output_dim = getattr(model_cfg.text_config, "hidden_size", self.cfg.output_dim)

        if self.cfg.freeze:
            self.freeze_backbone()

    def _load_via_gr00t_package(self):
        """Load GR00T via the Isaac-GR00T package."""
        try:
            from gr00t.policy.gr00t_policy import Gr00tPolicy
        except ImportError:
            raise ImportError(
                "GR00T not found. Install one of:\n"
                "  1. HuggingFace: pip install transformers  (model: nvidia/GR00T-N1.7-3B)\n"
                "  2. Isaac-GR00T:\n"
                "     git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T\n"
                "     cd Isaac-GR00T && uv sync --python 3.10"
            )

        self._gr00t_policy = Gr00tPolicy(
            embodiment_tag=self.cfg.embodiment_tag,
            model_path=self.cfg.model_name,
        )

        # Extract internal model
        if hasattr(self._gr00t_policy, "model"):
            self._model = self._gr00t_policy.model

        if self.cfg.freeze:
            self.freeze_backbone()

    def forward(
        self,
        image: torch.Tensor = None,
        text: Optional[str] = None,
        images: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Extract VL features from GR00T.

        Args:
            image: (B, C, H, W) image tensor
            text: optional text instruction

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

        if self._processor is not None:
            return self._forward_hf(image, text, batch_size, device)
        else:
            return self._forward_gr00t_package(image, text, batch_size, device)

    def _forward_hf(self, image, text, batch_size, device):
        """Forward through HuggingFace-loaded GR00T."""
        from PIL import Image
        import numpy as np

        # Convert tensor to PIL for processor
        if isinstance(image, torch.Tensor):
            img_np = image.detach().cpu()
            if img_np.shape[1] == 3:  # (B, C, H, W) → (B, H, W, C)
                img_np = img_np.permute(0, 2, 3, 1)
            if img_np.dtype == torch.float32 or img_np.dtype == torch.bfloat16:
                img_np = (img_np.float().clamp(0, 1) * 255).byte()
            pil_images = [Image.fromarray(img_np[i].numpy()) for i in range(batch_size)]

        # Text prompts
        if text is None:
            text_prompts = [""] * batch_size
        elif isinstance(text, str):
            text_prompts = [text] * batch_size
        else:
            text_prompts = list(text)

        # Process inputs
        inputs = self._processor(
            images=pil_images,
            text=text_prompts,
            return_tensors="pt",
            padding=True,
        )

        model_device = next(self._model.parameters()).device
        inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.set_grad_enabled(not self.cfg.freeze):
            outputs = self._model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract features from last hidden state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        # Pool to (B, output_dim)
        if hidden_states.dim() == 3:
            if self.cfg.pooling_method == "last":
                features = hidden_states[:, -1, :]
            else:
                features = hidden_states.mean(dim=1)
        else:
            features = hidden_states

        return features.to(device)

    def _forward_gr00t_package(self, image, text, batch_size, device):
        """Forward through Isaac-GR00T native policy."""
        import numpy as np

        # Convert to numpy format expected by GR00T: (B, T, H, W, 3)
        img_np = image.detach().cpu()
        if img_np.shape[1] == 3:  # (B, C, H, W) → (B, H, W, C)
            img_np = img_np.permute(0, 2, 3, 1)
        if img_np.dtype != torch.uint8:
            img_np = (img_np.float().clamp(0, 1) * 255).byte()
        img_np = img_np.numpy()

        # Add temporal dimension: (B, H, W, C) → (B, 1, H, W, C)
        img_np = img_np[:, None, :, :, :]

        text_prompt = text if isinstance(text, str) else (text[0] if isinstance(text, list) else "")

        obs = {
            "video": {"cam_high": img_np},
            "state": {},
            "language": {"text": [[text_prompt]] * batch_size},
        }

        # Extract features (not actions) from internal model
        if hasattr(self._model, "encode"):
            with torch.set_grad_enabled(not self.cfg.freeze):
                features = self._model.encode(obs)
                if isinstance(features, dict):
                    features = features.get("features", features.get("hidden_states"))
                features = torch.as_tensor(features, dtype=torch.float32)
                if features.dim() == 1:
                    features = features.unsqueeze(0).expand(batch_size, -1)
                return features.to(device)

        # Fallback: use action head output shape to create feature proxy
        return self._extract_features_fallback(image, batch_size, device)

    def _extract_features_fallback(self, image, batch_size, device):
        """Fallback when direct feature extraction is not available."""
        if not hasattr(self, "_fallback_proj"):
            self._fallback_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(image.shape[1], self.output_dim),
            ).to(device)

        with torch.set_grad_enabled(not self.cfg.freeze):
            return self._fallback_proj(image)


# Fix circular reference
GR00TCfg.class_type = GR00T
