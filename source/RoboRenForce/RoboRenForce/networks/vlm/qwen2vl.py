"""
Qwen2-VL Backbone Wrapper

Wrapper for Qwen2-VL vision-language model from HuggingFace.

Reference: .references/Psi0/src/psi/models/qwen3vl_wrapper.py

Adaptations:
- Uses Qwen2-VL (Qwen3-VL not yet released as of implementation)
- Integrates with RoboRenForce configclass system
- Supports multiple model sizes (2B, 7B)
"""

from typing import Optional, Dict, Any
from dataclasses import MISSING
import warnings

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor

from RoboRenForce.utils.configclass import configclass
from .vlm_backbone_base import VLMBackbone, VLMBackboneCfg


class Qwen2VL(VLMBackbone):
    """
    Qwen2-VL wrapper for VLA.

    Loads pretrained Qwen2-VL and extracts VL features for action prediction.

    Features:
    - Supports Qwen2-VL-2B-Instruct and Qwen2-VL-7B-Instruct
    - Optional freezing (default: True for efficient training)
    - Optional LoRA fine-tuning via PEFT
    - Extracts pooled vision-language features
    """

    def __init__(self, cfg: "Qwen2VLCfg"):
        super().__init__(cfg)

        self.cfg = cfg

        print(f"Loading Qwen2-VL model: {cfg.model_name}")
        print(f"  Freeze: {cfg.freeze}")
        print(f"  Use LoRA: {cfg.use_lora}")

        # Load model with trust_remote_code (Qwen2-VL requires custom code)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float32,
            device_map="auto" if cfg.device_map_auto else None,
            trust_remote_code=True,
        )

        # Load processor (handles image preprocessing and tokenization)
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
        )

        # Get actual output dimension from model config
        self.output_dim = self.model.config.hidden_size
        if self.output_dim != cfg.output_dim:
            warnings.warn(
                f"Specified output_dim ({cfg.output_dim}) differs from model hidden_size ({self.output_dim}). "
                f"Using model's hidden_size: {self.output_dim}"
            )

        # Apply LoRA if specified
        if cfg.use_lora:
            self._apply_lora()

        # Freeze if specified
        if cfg.freeze:
            self.freeze_backbone()
            print(f"✓ Model frozen (System 2)")

        print(f"✓ Qwen2-VL loaded: output_dim={self.output_dim}")

    def _apply_lora(self):
        """Apply LoRA to the model for efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.cfg.lora_rank,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=self.cfg.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, lora_config)
            print(f"✓ LoRA applied: rank={self.cfg.lora_rank}, alpha={self.cfg.lora_alpha}")
            self.model.print_trainable_parameters()

        except ImportError:
            raise ImportError(
                "PEFT library not found. Install with: pip install peft"
            )

    def forward(
        self,
        images: torch.Tensor,
        text: Optional[str] = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """
        Extract VL features from Qwen2-VL.

        Args:
            images: (B, C, H, W) image tensor or list of PIL images
            text: Optional text prompt (string or list of strings for batch)
            return_dict: If True, return dict with additional info

        Returns:
            vl_features: (B, output_dim) VL feature tensor
        """
        # Handle batch size
        if isinstance(images, torch.Tensor):
            batch_size = images.shape[0]
        else:
            batch_size = len(images) if isinstance(images, list) else 1

        # Prepare text (use default if None)
        if text is None:
            text = ["Describe this image."] * batch_size
        elif isinstance(text, str):
            text = [text] * batch_size

        # Preprocess inputs
        # Note: Qwen2-VL processor expects PIL images or image paths
        # If we have tensors, we need to convert them
        if isinstance(images, torch.Tensor):
            # Convert tensor to PIL images (assuming normalized [0,1] or [-1,1])
            from PIL import Image
            import numpy as np

            # Denormalize if needed and convert to uint8
            images_np = images.cpu().numpy()
            if images_np.min() < 0:  # Assume [-1, 1] normalization
                images_np = (images_np + 1) / 2
            images_np = (images_np * 255).astype(np.uint8)

            # Convert to PIL (B, C, H, W) -> list of (H, W, C) PIL images
            pil_images = []
            for i in range(batch_size):
                img_np = images_np[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                pil_images.append(Image.fromarray(img_np))

            images = pil_images

        # Process inputs with Qwen2-VL processor
        inputs = self.processor(
            images=images,
            text=text,
            return_tensors="pt",
            padding=True,
        )

        # Move to model device
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Forward pass
        with torch.set_grad_enabled(not self.cfg.freeze):
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract features from last hidden state
        # Shape: (B, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states[-1]

        # Pool features (use last token or mean pooling)
        if self.cfg.pooling_method == "last":
            # Use last token
            vl_features = hidden_states[:, -1, :]
        elif self.cfg.pooling_method == "mean":
            # Mean pooling over sequence
            vl_features = hidden_states.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.cfg.pooling_method}")

        # Shape: (B, output_dim)
        assert vl_features.shape == (batch_size, self.output_dim), \
            f"Expected shape ({batch_size}, {self.output_dim}), got {vl_features.shape}"

        if return_dict:
            return {
                "features": vl_features,
                "hidden_states": hidden_states,
                "outputs": outputs,
            }

        return vl_features

    def freeze_backbone(self):
        """Freeze all VLM parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze_backbone(self):
        """Unfreeze VLM parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()


@configclass
class Qwen2VLCfg(VLMBackboneCfg):
    """
    Qwen2-VL configuration.

    Model variants:
    - Qwen/Qwen2-VL-2B-Instruct (2B parameters, fast)
    - Qwen/Qwen2-VL-7B-Instruct (7B parameters, better quality)
    """

    class_type: type[Qwen2VL] = Qwen2VL

    # Model
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    freeze: bool = True
    output_dim: int = 1536  # Qwen2-VL-2B hidden dim (will be auto-detected)

    # Device and dtype
    use_bf16: bool = True  # Use bfloat16 for efficiency
    device_map_auto: bool = True  # Auto device mapping for multi-GPU

    # Feature extraction
    pooling_method: str = "last"  # "last" or "mean"

    # LoRA (optional, for fine-tuning VLM)
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Auto-detect if None

    def __post_init__(self):
        # Set default LoRA target modules if not specified
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
