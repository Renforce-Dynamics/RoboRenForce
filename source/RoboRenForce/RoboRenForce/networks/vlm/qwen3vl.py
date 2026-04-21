"""
Qwen3-VL Backbone Wrapper

Wrapper for Qwen3-VL vision-language model from HuggingFace.
Follows the same pattern as Qwen2VL but uses Qwen3-VL API.

Differences from Qwen2-VL:
- Larger hidden dim (2048 for 2B vs 1536 for Qwen2-VL-2B)
- Updated architecture with improved vision encoder
- Uses Qwen2_5_VLForConditionalGeneration (HF class name for Qwen3-VL)
"""

from typing import Optional
import warnings

import torch
import torch.nn as nn
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None

from RoboRenForce.utils.configclass import configclass
from .vlm_backbone_base import VLMBackbone, VLMBackboneCfg


class Qwen3VL(VLMBackbone):
    """
    Qwen3-VL wrapper for VLA.

    Loads pretrained Qwen3-VL and extracts VL features for action prediction.
    Supports Qwen3-VL-2B-Instruct and Qwen3-VL-8B-Instruct.
    """

    def __init__(self, cfg: "Qwen3VLCfg"):
        super().__init__(cfg)
        self.cfg = cfg

        print(f"Loading Qwen3-VL model: {cfg.model_name}")
        print(f"  Freeze: {cfg.freeze}")
        print(f"  Use LoRA: {cfg.use_lora}")

        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError(
                "transformers >= 4.45 required for Qwen3-VL. "
                "Install with: pip install transformers>=4.45"
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float32,
            device_map="auto" if cfg.device_map_auto else None,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
        )

        # Auto-detect output dimension
        model_cfg = self.model.config
        if hasattr(model_cfg, 'hidden_size'):
            self.output_dim = model_cfg.hidden_size
        elif hasattr(model_cfg, 'text_config'):
            self.output_dim = model_cfg.text_config.hidden_size
        else:
            self.output_dim = cfg.output_dim

        if self.output_dim != cfg.output_dim:
            warnings.warn(
                f"Specified output_dim ({cfg.output_dim}) differs from model "
                f"hidden_size ({self.output_dim}). Using model's: {self.output_dim}"
            )

        if cfg.use_lora:
            self._apply_lora()

        if cfg.freeze:
            self.freeze_backbone()
            print(f"✓ Model frozen (System 2)")

        print(f"✓ Qwen3-VL loaded: output_dim={self.output_dim}")

    def _apply_lora(self):
        """Apply LoRA for efficient fine-tuning."""
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
            raise ImportError("PEFT required for LoRA. Install: pip install peft")

    def _tensor_to_pil(self, images: torch.Tensor) -> list:
        """Convert (B, C, H, W) tensor to list of PIL images."""
        from PIL import Image
        import numpy as np

        images_np = images.detach().cpu().float().numpy()
        if images_np.min() < 0:
            images_np = (images_np + 1) / 2
        images_np = (images_np.clip(0, 1) * 255).astype(np.uint8)
        return [Image.fromarray(images_np[i].transpose(1, 2, 0)) for i in range(images_np.shape[0])]

    def forward(
        self,
        images: torch.Tensor = None,
        text: Optional[str] = None,
        return_dict: bool = False,
        image: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Extract VL features from Qwen3-VL.

        Args:
            images: (B, C, H, W) image tensor
            text: optional text prompt (str or list[str])
            return_dict: if True, return dict with hidden_states

        Returns:
            vl_features: (B, output_dim)
        """
        if images is None:
            images = image
        if images is None:
            raise ValueError("Either 'images' or 'image' must be provided")

        if isinstance(images, torch.Tensor):
            batch_size = images.shape[0]
            pil_images = self._tensor_to_pil(images)
        else:
            pil_images = images if isinstance(images, list) else [images]
            batch_size = len(pil_images)

        if text is None:
            text_prompts = ["Describe this image."] * batch_size
        elif isinstance(text, str):
            text_prompts = [text] * batch_size
        else:
            text_prompts = list(text)

        # Build message format
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None

        all_texts = []
        all_image_inputs = []

        for i in range(batch_size):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_images[i]},
                    {"type": "text", "text": text_prompts[i]},
                ],
            }]
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_texts.append(chat_text)

            if process_vision_info is not None:
                img_inputs, _ = process_vision_info(messages)
                all_image_inputs.extend(img_inputs)
            else:
                all_image_inputs.append(pil_images[i])

        inputs = self.processor(
            text=all_texts,
            images=all_image_inputs,
            padding=True,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.set_grad_enabled(not self.cfg.freeze):
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states[-1]

        if self.cfg.pooling_method == "last":
            vl_features = hidden_states[:, -1, :]
        elif self.cfg.pooling_method == "mean":
            vl_features = hidden_states.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.cfg.pooling_method}")

        if return_dict:
            return {
                "features": vl_features,
                "hidden_states": hidden_states,
                "outputs": outputs,
            }

        return vl_features

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()


@configclass
class Qwen3VLCfg(VLMBackboneCfg):
    """
    Qwen3-VL configuration.

    Model variants:
    - Qwen/Qwen3-VL-2B-Instruct (2B params, output_dim=2048)
    - Qwen/Qwen3-VL-8B-Instruct (8B params)
    """

    class_type: type[Qwen3VL] = Qwen3VL

    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    freeze: bool = True
    output_dim: int = 2048

    use_bf16: bool = True
    device_map_auto: bool = True
    pooling_method: str = "last"

    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: list = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
