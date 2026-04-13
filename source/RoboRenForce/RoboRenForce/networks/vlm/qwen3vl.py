"""
Qwen3-VL Backbone Wrapper

Wrapper for Qwen3-VL vision-language model.

Reference: .references/Psi0/src/psi/models/qwen3vl_wrapper.py

TODO Phase 1.2 (Week 1, Priority P0):
- [ ] Load Qwen3-VL from HuggingFace
- [ ] Implement feature extraction
- [ ] Support different Qwen3-VL variants (2B, 8B, etc.)
- [ ] Add LoRA support (optional)
- [ ] Handle image preprocessing
- [ ] Handle text tokenization
"""

from typing import Optional

import torch
import torch.nn as nn

from RoboRenForce.utils.configclass import configclass, MISSING
from .vlm_backbone_base import VLMBackbone, VLMBackboneCfg


@configclass
class Qwen3VLCfg(VLMBackboneCfg):
    """
    Qwen3-VL configuration.

    Model variants:
    - Qwen/Qwen3-VL-2B-Instruct (2B parameters, fast)
    - Qwen/Qwen3-VL-8B-Instruct (8B parameters, better)
    """

    class_type: type["Qwen3VL"] = MISSING

    # Model
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    freeze: bool = True
    output_dim: int = 2048  # Qwen3-VL-2B hidden dim

    # LoRA (if fine-tuning VLM)
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = ["q_proj", "v_proj"]


class Qwen3VL(VLMBackbone):
    """
    Qwen3-VL wrapper for VLA.

    Loads pretrained Qwen3-VL and extracts VL features for action prediction.

    TODO Phase 1.2:
    - [ ] Load model from HuggingFace transformers
    - [ ] Setup image processor
    - [ ] Setup text tokenizer
    - [ ] Implement forward pass (extract features)
    - [ ] Add LoRA support via PEFT
    - [ ] Handle different model sizes
    """

    def __init__(self, cfg: Qwen3VLCfg):
        super().__init__(cfg)

        # TODO: Load Qwen3-VL model
        # from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        #
        # self.model = Qwen3VLForConditionalGeneration.from_pretrained(
        #     cfg.model_name,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        # )
        #
        # self.processor = AutoProcessor.from_pretrained(cfg.model_name)
        raise NotImplementedError("TODO: Load Qwen3-VL model")

        # TODO: Apply LoRA if specified
        # if cfg.use_lora:
        #     from peft import LoraConfig, get_peft_model
        #     lora_config = LoraConfig(
        #         r=cfg.lora_rank,
        #         lora_alpha=cfg.lora_alpha,
        #         lora_dropout=cfg.lora_dropout,
        #         target_modules=cfg.lora_target_modules,
        #     )
        #     self.model = get_peft_model(self.model, lora_config)

        # TODO: Freeze if specified
        # if cfg.freeze:
        #     self.freeze_backbone()

    def forward(
        self,
        image: torch.Tensor,
        text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract VL features from Qwen3-VL.

        Args:
            image: (B, C, H, W) image tensor
            text: (B, max_text_len) token IDs or None

        Returns:
            vl_features: (B, output_dim) VL feature tensor

        TODO:
        - Preprocess inputs with self.processor
        - Forward through self.model
        - Extract hidden states (before LM head)
        - Pool features (e.g., last token, mean pooling)
        - Return (B, output_dim) tensor
        """
        raise NotImplementedError("TODO: Implement Qwen3-VL forward pass")

        # Example structure:
        # # Preprocess
        # inputs = self.processor(images=image, text=text, return_tensors="pt")
        # inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        #
        # # Forward
        # with torch.set_grad_enabled(not self.cfg.freeze):
        #     outputs = self.model.model(**inputs)  # Base model (no LM head)
        #     hidden_states = outputs.last_hidden_state  # (B, seq_len, hidden_dim)
        #
        # # Pool features (take last token)
        # vl_features = hidden_states[:, -1, :]  # (B, hidden_dim)
        #
        # return vl_features
