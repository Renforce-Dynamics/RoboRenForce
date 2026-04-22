# RoboRenForce Core Framework Structure

**Root**: `source/RoboRenForce/RoboRenForce/`

This document details the main framework codebase structure and planned VLA integration.

---

## Directory Overview

```
RoboRenForce/
├── algorithms/          # Training algorithms (PPO, SAC, GAIL, etc.)
├── buffer/             # Data storage and replay systems
├── components/         # Modular components (actors, critics, encoders, etc.)
├── dataset/           # Dataset loaders and wrappers
├── networks/          # Neural network primitives (MLP, Transformer, etc.)
├── runners/           # Training orchestration (on-policy, off-policy, VLA)
└── utils/             # Utilities (configclass, env wrappers, logging, etc.)
```

---

## 1. Algorithms: `algorithms/`

### 1.1 Current Structure (RL-only)

```
algorithms/
├── algorithm_base.py              # Base class for all algorithms
├── on_policy/
│   ├── ppo.py                    # Proximal Policy Optimization
│   ├── sacp.py                   # SAC with policy head
│   ├── smooth.py                 # Smooth policy regularization
│   ├── epo/                      # Experience-based Policy Optimization
│   ├── mbpo/                     # Model-Based Policy Optimization
│   └── sapg/                     # Stochastic Analytical Policy Gradient
├── off_policy/
│   ├── sac/                      # Soft Actor-Critic
│   └── dsac/                     # Distributional SAC
├── imitation/
│   ├── distillation.py           # Policy distillation
│   └── adversarial/              # GAIL and variants
├── smooth/
│   ├── CAPS.py                   # Constraint-based smoothing
│   ├── L2C2.py                   # Lipschitz control
│   └── Lips.py                   # Lipschitz regularization
└── nn_model_trainer/
    ├── flow_model_trainer.py     # Flow-based dynamics
    └── system_dynamics_trainer.py # System identification
```

### 1.2 VLA Extensions (New)

**Phase 5: Post-Train Algorithms** (Week 5, Priority P1)

```
algorithms/
└── vla_training/                 # VLA-specific training algorithms
    ├── __init__.py
    ├── pretrain_algorithm.py     # Supervised pretraining on offline data
    ├── sft_algorithm.py          # Supervised Fine-Tuning
    ├── dpo_algorithm.py          # Direct Preference Optimization (Future)
    └── rl_finetune_algorithm.py  # RL fine-tuning (PPO/SAC on VLA)
```

**File: `pretrain_algorithm.py`** (Phase 3, Week 3, Priority P0)
```python
from RoboRenForce.utils.configclass import configclass, MISSING
from RoboRenForce.algorithms.algorithm_base import AlgorithmBaseCfg, AlgorithmBase

@configclass
class VLAPretrainAlgorithmCfg(AlgorithmBaseCfg):
    """Supervised pretraining for VLA Action Expert."""
    class_type: type['VLAPretrainAlgorithm'] = MISSING
    
    # Loss weights
    action_loss_weight: float = 1.0
    kl_loss_weight: float = 0.0  # If using variational action head
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bf16"  # "fp16" or "bf16"

class VLAPretrainAlgorithm(AlgorithmBase):
    """
    Supervised pretraining algorithm for VLA.
    
    Loss: L2 on action predictions
    Supports: Diffusion action heads, MLP action heads
    
    Reference: .references/Psi0/src/psi/training/pretrain.py
    """
    
    def __init__(self, cfg: VLAPretrainAlgorithmCfg):
        super().__init__(cfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    
    def compute_loss(self, batch, vla_actor):
        """
        Args:
            batch: Dict with keys [image, text, proprioception, action]
            vla_actor: VLA actor module
        
        Returns:
            loss_dict: {total_loss, action_loss, ...}
        """
        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp, dtype=torch.bfloat16):
            # Forward pass
            pred_action = vla_actor(batch)
            
            # Action loss (L2 or log-likelihood)
            action_loss = F.mse_loss(pred_action, batch["action"])
            
            # Total loss
            total_loss = self.cfg.action_loss_weight * action_loss
        
        return {
            "total_loss": total_loss,
            "action_loss": action_loss,
        }
    
    def update(self, batch, vla_actor, optimizer):
        """Single optimization step."""
        loss_dict = self.compute_loss(batch, vla_actor)
        
        optimizer.zero_grad()
        self.scaler.scale(loss_dict["total_loss"]).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vla_actor.parameters(), self.cfg.max_grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss_dict
```

**File: `rl_finetune_algorithm.py`** (Phase 6, Week 6, Priority P1)
```python
@configclass
class VLARLFinetuneAlgorithmCfg(AlgorithmBaseCfg):
    """RL fine-tuning algorithm for VLA (PPO or SAC)."""
    class_type: type['VLARLFinetuneAlgorithm'] = MISSING
    
    # Base RL algorithm
    base_rl_algorithm_cfg: AlgorithmBaseCfg = MISSING  # PPOCfg or SACCfg
    
    # LoRA fine-tuning
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    
    # Freeze VLM backbone
    freeze_vlm: bool = True

class VLARLFinetuneAlgorithm(AlgorithmBase):
    """
    RL fine-tuning for pretrained VLA.
    
    Wraps PPO/SAC algorithm with LoRA support.
    Reference: .references/RoboTwin/lora_finetuning.py
    """
    
    def __init__(self, cfg: VLARLFinetuneAlgorithmCfg):
        super().__init__(cfg)
        # Construct base RL algorithm (PPO or SAC)
        self.base_algorithm = cfg.base_rl_algorithm_cfg.construct_from_cfg()
```

---

## 2. Buffers: `buffer/`

### 2.1 Current Structure (RL Buffers)

```
buffer/
├── replay_buffer_base.py         # Base class
├── replay_bundle.py              # Bundle of multiple buffers
├── online_rollout/               # On-policy rollout storage
│   ├── rollout_storage.py       # Standard PPO storage
│   ├── rollout_storage_multi.py # Multi-env storage
│   ├── belief_rollout_storage.py
│   └── sapg_rollout_storage.py
├── direct_based/                 # Off-policy replay buffers
│   ├── dynamic_replay_buffer.py
│   ├── flow_replay_buffer.py
│   └── transition_buffer/
└── pipeline_based/               # Pipeline-style buffers
    ├── data_pipeline.py
    ├── pipe_buffer_base.py
    ├── pipe_buffer_onpolicy.py
    ├── pipe_buffer_chunk/
    └── pipe_buffer_transition/
```

### 2.2 VLA Extensions (Future - Not in MVP)

VLA uses **offline datasets** for pretraining, so no new buffer types needed for Phase 3-5.

**Phase 6 (RL Fine-tuning)** will reuse existing `rollout_storage.py` for on-policy VLA+RL training.

---

## 3. Components: `components/`

### 3.1 Current Structure (RL Components)

```
components/
├── actor/                        # Policy networks
│   ├── actor_base.py
│   ├── gaussian_actor.py        # Standard Gaussian policy
│   ├── sac_actor.py             # SAC actor
│   ├── lipschitz_actor.py       # Smooth policies
│   ├── encoder_state_actor.py   # With state encoder
│   └── belief_encoder_actor.py  # Belief-based policies
├── critic/                       # Value networks
│   ├── q_network.py
│   ├── v_network.py
│   ├── multi_q_network.py       # Ensemble Q
│   └── distributional_q_network.py
├── actor_critic_pack/
│   └── actor_critic_pack.py     # Bundle actor + critic
├── encoder/
│   └── vec_state_encoder.py    # State feature extraction
├── decoder/
│   ├── continuous_vec_decoder.py
│   └── transition_vec_decoder.py
├── discriminator/                # For GAIL
│   └── discriminator.py
├── normalizer/
│   ├── normalizer_base.py
│   ├── normalizer_empirical.py
│   └── running_scale.py
├── nn_models/
│   ├── nn_model_base.py
│   ├── system_dynamics/
│   ├── belief_flow_model/
│   └── tdmpcs/
└── wrapper/
    ├── module_dict.py
    └── module_list.py
```

### 3.2 VLA Components (New)

**Phase 2: Action Expert** (Week 2, Priority P0)

```
components/
└── actor/
    ├── vla_actor.py              # VLA actor (System 1 + System 2 fusion) ⭐
    └── action_heads/             # Action Expert heads
        ├── __init__.py
        ├── diffusion_action_head.py    # Diffusion Transformer head ⭐
        └── regression_action_head.py   # Simple MLP head (baseline)
```

**File: `vla_actor.py`** (Phase 2, Week 2, Priority P0)
```python
from RoboRenForce.utils.configclass import configclass, MISSING
from RoboRenForce.components.actor.actor_base import ActorBaseCfg, ActorBase

@configclass
class VLAActorCfg(ActorBaseCfg):
    """
    VLA Actor combining:
    - System 2: VLM backbone (frozen, for VL feature extraction)
    - Fusion: Proprioception integration
    - System 1: Action Expert (trainable)
    """
    class_type: type['VLAActor'] = MISSING
    
    # System 2: VLM backbone
    vlm_backbone_cfg: VLMBackboneCfg = MISSING  # From networks/vlm/
    freeze_vlm: bool = True
    
    # Fusion layer
    fusion_cfg: FusionLayerCfg = FusionLayerCfg()
    
    # System 1: Action Expert
    action_head_cfg: ActionHeadCfg = MISSING  # Diffusion or Regression
    
    # Input modalities
    use_proprioception: bool = True
    use_text: bool = True

class VLAActor(ActorBase):
    """
    VLA Actor implementation.
    
    Architecture:
    1. VLM backbone extracts vision-language features (frozen System 2)
    2. Fusion layer combines VL features + proprioception
    3. Action head predicts actions (trainable System 1)
    
    Reference: .references/Psi0/src/psi/models/psi0.py
    """
    
    def __init__(self, cfg: VLAActorCfg, dim_params: dict):
        super().__init__(cfg, dim_params)
        
        # System 2: VLM backbone
        self.vlm = cfg.vlm_backbone_cfg.construct_from_cfg(dim_params)
        if cfg.freeze_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
        
        # Fusion layer
        fusion_dim_params = {
            "vl_feature_dim": self.vlm.output_dim,
            "proprio_dim": dim_params.get("proprioception_dim", 0),
        }
        self.fusion = cfg.fusion_cfg.construct_from_cfg(fusion_dim_params)
        
        # System 1: Action Expert
        action_dim_params = {
            "input_dim": self.fusion.output_dim,
            "action_dim": dim_params["action_dim"],
        }
        self.action_head = cfg.action_head_cfg.construct_from_cfg(action_dim_params)
    
    def forward(self, obs_dict: dict, deterministic: bool = False):
        """
        Args:
            obs_dict: {
                "image": (B, C, H, W),
                "text": (B, max_text_len),  # Optional
                "proprioception": (B, proprio_dim),  # Optional
            }
        
        Returns:
            action: (B, action_dim)
        """
        # System 2: VLM features (no grad if frozen)
        with torch.set_grad_enabled(not self.cfg.freeze_vlm):
            vl_features = self.vlm(
                image=obs_dict["image"],
                text=obs_dict.get("text", None),
            )
        
        # Fusion
        if self.cfg.use_proprioception:
            fused_features = self.fusion(vl_features, obs_dict["proprioception"])
        else:
            fused_features = vl_features
        
        # System 1: Action prediction
        action = self.action_head(fused_features, deterministic=deterministic)
        
        return action
```

**File: `action_heads/diffusion_action_head.py`** (Phase 2, Week 2, Priority P0)
```python
@configclass
class DiffusionActionHeadCfg(ModuleBaseCfg):
    """
    Diffusion Transformer action head.
    
    Predicts action via denoising diffusion.
    Reference: .references/Psi0/src/psi/models/action_head.py
    """
    class_type: type['DiffusionActionHead'] = MISSING
    
    # Transformer architecture
    num_layers: int = 4
    num_heads: int = 8
    embed_dim: int = 256
    
    # Diffusion parameters
    num_diffusion_steps: int = 10  # DDIM sampling steps
    noise_schedule: str = "cosine"  # "linear" or "cosine"
    
    # Action chunking
    action_horizon: int = 1  # Future actions to predict
    action_dim: int = MISSING

class DiffusionActionHead(ModuleBase):
    """
    Diffusion-based action head using DDIM sampling.
    
    Training: Denoise actions from Gaussian noise
    Inference: Iterative denoising to generate actions
    """
    
    def __init__(self, cfg: DiffusionActionHeadCfg, dim_params: dict):
        super().__init__(cfg)
        
        # Noise prediction network (Transformer)
        self.noise_predictor = TransformerBackbone(
            input_dim=cfg.action_dim + dim_params["input_dim"],
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            embed_dim=cfg.embed_dim,
        )
        
        # Noise schedule
        self.register_noise_schedule(cfg.noise_schedule, cfg.num_diffusion_steps)
    
    def forward(self, features, deterministic: bool = False):
        """
        Args:
            features: (B, input_dim) from fusion layer
            deterministic: If True, use DDIM sampling; else, sample once
        
        Returns:
            action: (B, action_horizon, action_dim)
        """
        if self.training:
            # Training: denoise from noisy actions
            return self.train_forward(features)
        else:
            # Inference: DDIM sampling
            return self.sample_actions(features, num_steps=self.cfg.num_diffusion_steps)
    
    def train_forward(self, features):
        """Compute denoising loss during training."""
        # Sample noise and timestep
        # Predict noise with Transformer
        # Return predicted noise (loss computed in algorithm)
        pass
    
    def sample_actions(self, features, num_steps: int):
        """DDIM sampling for action generation."""
        # Start from Gaussian noise
        # Iteratively denoise using noise_predictor
        # Return final denoised action
        pass
```

**File: `action_heads/regression_action_head.py`** (Phase 2, Week 2, Priority P1)
```python
@configclass
class RegressionActionHeadCfg(ModuleBaseCfg):
    """Simple MLP action head (baseline)."""
    class_type: type['RegressionActionHead'] = MISSING
    
    hidden_dims: list[int] = [256, 256]
    activation: str = "relu"
    action_dim: int = MISSING

class RegressionActionHead(ModuleBase):
    """MLP-based action head for direct regression."""
    
    def __init__(self, cfg: RegressionActionHeadCfg, dim_params: dict):
        super().__init__(cfg)
        self.mlp = MLP(
            input_dim=dim_params["input_dim"],
            output_dim=cfg.action_dim,
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
        )
    
    def forward(self, features, deterministic: bool = False):
        return self.mlp(features)
```

---

## 4. Dataset: `dataset/`

### 4.1 Current Structure (RL Datasets)

```
dataset/
├── data_loader/
│   ├── offline_data_loader_base.py
│   ├── hdf5_data_loader_base.py
│   └── nn_model/
└── data_wrapper/
    ├── offline_data_wrapper_base.py
    └── nn_model_data_wrapper.py
```

### 4.2 VLA Datasets (New)

**Phase 1: Data Infrastructure** (Week 1, Priority P0)

```
dataset/
├── lerobot/                      # LeRobot format support ⭐
│   ├── __init__.py
│   ├── lerobot_dataset.py       # Main dataset class ⭐
│   ├── lerobot_processor.py     # Data preprocessing pipeline ⭐
│   └── dataset_stats.py         # Stats computation/loading
└── mixture/                      # Mixed dataset sampling (Future)
    ├── __init__.py
    └── mixture_dataset.py       # Sample from multiple datasets
```

**File: `lerobot_dataset.py`** (Phase 1.1, Week 1, Priority P0)
```python
from torch.utils.data import Dataset
from RoboRenForce.utils.configclass import configclass, MISSING

@configclass
class LeRobotDatasetCfg(ModuleBaseCfg):
    """
    LeRobot dataset config.
    
    Data format: Parquet files with episode structure
    Reference: .references/lerobot/lerobot/common/datasets/lerobot_dataset.py
    """
    class_type: type['LeRobotDataset'] = MISSING
    
    # Data paths
    data_root: str = MISSING  # Path to dataset directory
    split: str = "train"  # "train" or "val"
    
    # Processing
    processor_cfg: LeRobotProcessorCfg = LeRobotProcessorCfg()
    
    # Performance
    load_videos: bool = True
    video_backend: str = "pyav"  # "pyav" or "opencv"
    num_workers: int = 4

class LeRobotDataset(Dataset):
    """
    PyTorch dataset for LeRobot format.
    
    Structure:
    - data/
      - meta/
        - stats.safetensors
        - info.json
      - train/
        - episode_000000.parquet
        - videos/
          - episode_000000_camera_0.mp4
    """
    
    def __init__(self, cfg: LeRobotDatasetCfg):
        super().__init__()
        self.cfg = cfg
        
        # Load metadata
        self.load_metadata()
        
        # Load episode parquet files
        self.load_episodes()
        
        # Initialize processor
        self.processor = cfg.processor_cfg.construct_from_cfg()
    
    def __getitem__(self, idx):
        """
        Returns:
            {
                "image": (C, H, W),
                "text": str or token_ids,
                "proprioception": (proprio_dim,),
                "action": (action_dim,),
                "reward": float,
                "done": bool,
            }
        """
        # Load raw data from parquet
        raw_data = self.get_raw_data(idx)
        
        # Process with processor (image transform, normalization, etc.)
        processed_data = self.processor(raw_data)
        
        return processed_data
```

**File: `lerobot_processor.py`** (Phase 1.1, Week 1, Priority P0)
```python
@configclass
class LeRobotProcessorCfg(ModuleBaseCfg):
    """
    Data preprocessing pipeline.
    
    Reference: .references/lerobot/lerobot/common/datasets/transforms.py
    """
    class_type: type['LeRobotProcessor'] = MISSING
    
    # Image processing
    image_size: tuple[int, int] = (224, 224)
    image_mean: list[float] = [0.485, 0.456, 0.406]
    image_std: list[float] = [0.229, 0.224, 0.225]
    
    # Normalization (from stats.safetensors)
    normalize_actions: bool = True
    normalize_proprioception: bool = True

class LeRobotProcessor:
    """Applies transforms to raw dataset samples."""
    
    def __init__(self, cfg: LeRobotProcessorCfg):
        self.cfg = cfg
        self.setup_transforms()
    
    def __call__(self, raw_data: dict) -> dict:
        """Process raw data sample."""
        processed = {}
        
        # Image: resize, normalize
        processed["image"] = self.image_transform(raw_data["image"])
        
        # Text: tokenize (if using text)
        if "text" in raw_data:
            processed["text"] = self.text_transform(raw_data["text"])
        
        # Proprioception: normalize
        if "proprioception" in raw_data:
            processed["proprioception"] = self.normalize(
                raw_data["proprioception"], "proprioception"
            )
        
        # Action: normalize
        processed["action"] = self.normalize(raw_data["action"], "action")
        
        return processed
```

---

## 5. Networks: `networks/`

### 5.1 Current Structure (RL Networks)

```
networks/
├── mlp.py                        # Multi-layer perceptron
├── activations.py
├── optimizer.py
├── conv2d.py
├── fft_filter.py
├── moe.py                        # Mixture of Experts
├── simple_rnn.py
├── transformer/
│   ├── transformer_backbone.py
│   └── structures/
└── vae/
    ├── mlp_vae.py
    └── vqvae.py
```

### 5.2 VLA Networks (New)

**Phase 1: VLM Backbone Loading** (Week 1, Priority P0)

```
networks/
└── vlm/                          # Vision-Language Models ⭐
    ├── __init__.py
    ├── vlm_backbone_base.py     # Base class for VLM backbones
    ├── qwen3vl.py               # Qwen3-VL wrapper ⭐
    ├── paligemma.py             # PaliGemma wrapper (Future)
    └── fusion_layers.py         # VL + Proprioception fusion ⭐
```

**File: `vlm/vlm_backbone_base.py`** (Phase 1.2, Week 1, Priority P0)
```python
from RoboRenForce.utils.configclass import configclass, MISSING
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg

@configclass
class VLMBackboneCfg(ModuleBaseCfg):
    """Base config for VLM backbones."""
    class_type: type['VLMBackbone'] = MISSING
    
    model_name: str = MISSING
    freeze: bool = True
    output_dim: int = MISSING  # VL feature dimension

class VLMBackbone(ModuleBase):
    """
    Base class for VLM backbones.
    
    Responsibilities:
    - Load pretrained VLM from HuggingFace
    - Extract vision-language features
    - Support freezing
    """
    
    def __init__(self, cfg: VLMBackboneCfg):
        super().__init__(cfg)
        self.output_dim = cfg.output_dim
    
    def forward(self, image, text=None):
        """
        Args:
            image: (B, C, H, W)
            text: (B, max_text_len) or None
        
        Returns:
            vl_features: (B, output_dim)
        """
        raise NotImplementedError
```

**File: `vlm/qwen3vl.py`** (Phase 1.2, Week 1, Priority P0)
```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

@configclass
class Qwen3VLCfg(VLMBackboneCfg):
    """
    Qwen3-VL backbone config.
    
    Reference: .references/Psi0/src/psi/models/qwen3vl_wrapper.py
    """
    class_type: type['Qwen3VL'] = MISSING
    
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    freeze: bool = True
    output_dim: int = 2048  # Qwen3-VL hidden dim
    
    # LoRA (if fine-tuning VLM)
    use_lora: bool = False
    lora_rank: int = 8
    lora_target_modules: list[str] = ["q_proj", "v_proj"]

class Qwen3VL(VLMBackbone):
    """
    Qwen3-VL wrapper.
    
    Loads pretrained Qwen3-VL and extracts VL features.
    """
    
    def __init__(self, cfg: Qwen3VLCfg):
        super().__init__(cfg)
        
        # Load model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Processor (image + text)
        self.processor = AutoProcessor.from_pretrained(cfg.model_name)
        
        # Apply LoRA if needed
        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                target_modules=cfg.lora_target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
        
        # Freeze if specified
        if cfg.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, image, text=None):
        """
        Extract VL features from Qwen3-VL.
        
        Returns:
            vl_features: (B, output_dim)
        """
        # Preprocess inputs
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Extract features (before language modeling head)
        with torch.no_grad() if self.cfg.freeze else torch.enable_grad():
            outputs = self.model.model(**inputs)  # Base model outputs
            # Pool or select last hidden state
            vl_features = outputs.last_hidden_state[:, -1, :]  # (B, hidden_dim)
        
        return vl_features
```

**File: `vlm/fusion_layers.py`** (Phase 2, Week 2, Priority P0)
```python
@configclass
class FusionLayerCfg(ModuleBaseCfg):
    """
    Fusion layer for VL features + proprioception.
    
    Concatenates and projects to common dimension.
    """
    class_type: type['FusionLayer'] = MISSING
    
    fusion_type: str = "concat_mlp"  # "concat_mlp" or "cross_attention"
    output_dim: int = 512
    hidden_dims: list[int] = [512]

class FusionLayer(ModuleBase):
    """Fuses VL features with proprioception."""
    
    def __init__(self, cfg: FusionLayerCfg, dim_params: dict):
        super().__init__(cfg)
        
        input_dim = dim_params["vl_feature_dim"] + dim_params["proprio_dim"]
        
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=cfg.output_dim,
            hidden_dims=cfg.hidden_dims,
        )
        
        self.output_dim = cfg.output_dim
    
    def forward(self, vl_features, proprioception):
        """
        Args:
            vl_features: (B, vl_dim)
            proprioception: (B, proprio_dim)
        
        Returns:
            fused: (B, output_dim)
        """
        concat = torch.cat([vl_features, proprioception], dim=-1)
        return self.mlp(concat)
```

---

## 6. Runners: `runners/`

### 6.1 Current Structure (RL Runners)

```
runners/
├── base_runner.py                # Base runner class
├── on_policy/
│   ├── on_policy_runner.py      # PPO, etc.
│   └── massive_parallel/        # Multi-env parallelism
├── off_policy/
│   └── off_policy_runner.py     # SAC, etc.
├── offline/
│   ├── offline_runner_base.py
│   └── nn_model/
├── imitation/
│   ├── distillation_runner.py
│   └── adversarial/
├── nn_model/
│   ├── flow_model_runner.py
│   ├── nn_model_runner.py
│   └── mbpo_on_policy_runner.py
└── logger/
    ├── logger_base.py
    ├── tqdm_style_logger.py
    └── utils/
```

### 6.2 VLA Runners (New)

**Phase 3-6: VLA Training Runners** (Weeks 3-6, Priority P0/P1)

```
runners/
└── vla/                          # VLA training runners ⭐
    ├── __init__.py
    ├── pretrain/                 # Pretraining runners
    │   ├── __init__.py
    │   ├── vla_pretrain_runner.py           # Single-GPU pretrain ⭐
    │   └── vla_pretrain_runner_distributed.py  # DDP pretrain ⭐
    ├── post_train/               # Post-training (SFT/DPO)
    │   ├── __init__.py
    │   ├── vla_sft_runner.py
    │   └── vla_dpo_runner.py    # Future
    └── rl/                       # RL fine-tuning
        ├── __init__.py
        └── vla_rl_runner.py     # VLA + RL ⭐
```

**File: `vla/pretrain/vla_pretrain_runner.py`** (Phase 3, Week 3, Priority P0)
```python
from RoboRenForce.runners.base_runner import BaseRunner, BaseRunnerCfg
from RoboRenForce.utils.configclass import configclass, MISSING

@configclass
class VLAPretrainRunnerCfg(BaseRunnerCfg):
    """Single-GPU VLA pretraining runner."""
    class_type: type['VLAPretrainRunner'] = MISSING
    
    # Components
    vla_actor_cfg: VLAActorCfg = MISSING
    algorithm_cfg: VLAPretrainAlgorithmCfg = VLAPretrainAlgorithmCfg()
    dataset_cfg: LeRobotDatasetCfg = MISSING
    
    # Training
    batch_size: int = 32
    num_epochs: int = 10
    num_workers: int = 4
    
    # Checkpointing
    save_interval: int = 1000  # steps
    checkpoint_dir: str = "checkpoints/"

class VLAPretrainRunner(BaseRunner):
    """
    Single-GPU VLA pretraining runner.
    
    Pipeline:
    1. Load dataset (LeRobot format)
    2. Create VLA actor
    3. Train with supervised loss
    4. Save checkpoints
    
    Reference: .references/Psi0/src/psi/training/pretrain_trainer.py
    """
    
    def __init__(self, cfg: VLAPretrainRunnerCfg, log_dir: str, device: str = "cuda"):
        super().__init__(cfg, log_dir, device)
        
        # Dataset
        self.dataset = cfg.dataset_cfg.construct_from_cfg()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
        )
        
        # VLA Actor
        dim_params = self.get_dim_params()
        self.vla_actor = cfg.vla_actor_cfg.construct_from_cfg(dim_params)
        self.vla_actor.to(device)
        
        # Algorithm
        self.algorithm = cfg.algorithm_cfg.construct_from_cfg()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.vla_actor.parameters(),
            lr=self.algorithm.cfg.learning_rate,
            weight_decay=self.algorithm.cfg.weight_decay,
        )
        
        self.global_step = 0
    
    def learn(self, num_epochs: int):
        """Main training loop."""
        for epoch in range(num_epochs):
            for batch in self.dataloader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Update
                loss_dict = self.algorithm.update(batch, self.vla_actor, self.optimizer)
                
                # Log
                self.logger.log_scalars(loss_dict, self.global_step)
                
                # Save checkpoint
                if self.global_step % self.cfg.save_interval == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
    
    def save_checkpoint(self):
        """Save VLA actor checkpoint."""
        ckpt_path = f"{self.cfg.checkpoint_dir}/vla_step_{self.global_step}.pth"
        torch.save({
            "vla_actor": self.vla_actor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
```

**File: `vla/pretrain/vla_pretrain_runner_distributed.py`** (Phase 4, Week 4, Priority P0)
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

@configclass
class DistributedVLAPretrainRunnerCfg(VLAPretrainRunnerCfg):
    """Multi-GPU DDP VLA pretraining runner."""
    class_type: type['DistributedVLAPretrainRunner'] = MISSING
    
    # DDP settings
    backend: str = "nccl"
    find_unused_parameters: bool = False

class DistributedVLAPretrainRunner(VLAPretrainRunner):
    """
    Multi-GPU DDP VLA pretraining.
    
    Usage:
        torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py
    
    Reference: .references/lerobot/lerobot/scripts/train.py (DDP section)
    """
    
    def __init__(self, cfg: DistributedVLAPretrainRunnerCfg, log_dir: str, device: str):
        # Init distributed
        dist.init_process_group(backend=cfg.backend)
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{self.local_rank}"
        
        super().__init__(cfg, log_dir, device)
        
        # Wrap actor with DDP
        self.vla_actor = DDP(
            self.vla_actor,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=cfg.find_unused_parameters,
        )
        
        # Use DistributedSampler
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            sampler=self.sampler,
            num_workers=cfg.num_workers,
        )
    
    def learn(self, num_epochs: int):
        """DDP training loop."""
        for epoch in range(num_epochs):
            self.sampler.set_epoch(epoch)  # Shuffle per epoch
            
            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Update (gradients synced across GPUs)
                loss_dict = self.algorithm.update(batch, self.vla_actor, self.optimizer)
                
                # Log only on rank 0
                if self.local_rank == 0:
                    self.logger.log_scalars(loss_dict, self.global_step)
                
                # Save on rank 0 only
                if self.local_rank == 0 and self.global_step % self.cfg.save_interval == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
        
        dist.destroy_process_group()
```

**File: `vla/rl/vla_rl_runner.py`** (Phase 6, Week 6, Priority P1)
```python
@configclass
class VLARLRunnerCfg(BaseRunnerCfg):
    """RL fine-tuning runner for pretrained VLA."""
    class_type: type['VLARLRunner'] = MISSING
    
    # Pretrained VLA
    pretrained_vla_path: str = MISSING
    
    # VLA config (for loading)
    vla_actor_cfg: VLAActorCfg = MISSING
    
    # RL algorithm
    rl_algorithm_cfg: VLARLFinetuneAlgorithmCfg = MISSING
    
    # Environment
    task: str = MISSING
    num_envs: int = 4096

class VLARLRunner(BaseRunner):
    """
    RL fine-tuning for pretrained VLA.
    
    Pipeline:
    1. Load pretrained VLA
    2. Apply LoRA (optional)
    3. Run RL (PPO/SAC) in Isaac Lab environment
    4. Fine-tune action head + fusion layer
    
    Reference: .references/RoboTwin/lora_finetuning.py
    """
    
    def __init__(self, cfg: VLARLRunnerCfg, log_dir: str, device: str = "cuda"):
        super().__init__(cfg, log_dir, device)
        
        # Load pretrained VLA
        self.vla_actor = self.load_pretrained_vla(cfg.pretrained_vla_path)
        
        # Apply LoRA
        if cfg.rl_algorithm_cfg.use_lora:
            self.apply_lora(self.vla_actor)
        
        # Create RL environment (Isaac Lab)
        self.env = self.create_env(cfg.task, cfg.num_envs)
        
        # RL algorithm
        self.rl_algorithm = cfg.rl_algorithm_cfg.construct_from_cfg()
    
    def learn(self, num_iterations: int):
        """RL training loop."""
        # Similar to on_policy_runner.py
        # Collect rollouts with VLA actor
        # Update with PPO/SAC
        pass
```

---

## 7. Utils: `utils/`

### 7.1 Current Structure

```
utils/
├── configclass/                  # Config decorator system ⭐
│   ├── configclass.py           # Main decorator
│   ├── dict.py                  # Dict conversion
│   └── string.py                # String serialization
├── env_wrapper/
│   ├── gym_wrapper/
│   ├── lab_wrapper/
│   └── vec_env.py
├── isaaclab/
│   ├── envs/
│   ├── get_info.py
│   └── trajectory.py
├── gym/
│   ├── config.py
│   └── __init__.py
├── template/
│   ├── module_base.py           # ModuleBase class ⭐
│   └── class_template_base.py
├── argtool.py
├── logging.py
├── mapping.py
├── math.py
├── normalizer.py
└── package.py
```

### 7.2 VLA Utils (New)

**Phase 1-4: VLA Utilities** (Weeks 1-4, Priority P1)

```
utils/
└── vla/                          # VLA-specific utilities
    ├── __init__.py
    ├── processor/                # Data processing (moved from dataset)
    │   ├── __init__.py
    │   ├── image_processor.py   # Image transforms
    │   └── text_processor.py    # Text tokenization
    ├── distributed/              # DDP helpers
    │   ├── __init__.py
    │   ├── ddp_utils.py         # DDP setup/cleanup
    │   └── checkpoint_utils.py  # Distributed checkpoint saving
    └── video/                    # Video I/O
        ├── __init__.py
        ├── video_reader.py      # Efficient video decoding
        └── video_writer.py      # Video encoding for demos
```

---

## Priority Summary

### Phase 1 (Week 1): Data + VLM - **8 files**
**Priority P0 (Critical Path)**:
- `dataset/lerobot/lerobot_dataset.py`
- `dataset/lerobot/lerobot_processor.py`
- `networks/vlm/vlm_backbone_base.py`
- `networks/vlm/qwen3vl.py`

**Priority P1**:
- `dataset/lerobot/dataset_stats.py`
- `utils/vla/processor/image_processor.py`
- `utils/vla/processor/text_processor.py`
- `utils/vla/video/video_reader.py`

### Phase 2 (Week 2): Action Expert - **5 files**
**Priority P0**:
- `components/actor/vla_actor.py`
- `components/actor/action_heads/diffusion_action_head.py`
- `networks/vlm/fusion_layers.py`

**Priority P1**:
- `components/actor/action_heads/regression_action_head.py`

### Phase 3 (Week 3): Single-GPU Training - **6 files**
**Priority P0**:
- `algorithms/vla_training/pretrain_algorithm.py`
- `runners/vla/pretrain/vla_pretrain_runner.py`

**Priority P1**:
- Evaluation utilities

### Phase 4 (Week 4): DDP Training - **8 files** ⭐
**Priority P0**:
- `runners/vla/pretrain/vla_pretrain_runner_distributed.py`
- `utils/vla/distributed/ddp_utils.py`
- `utils/vla/distributed/checkpoint_utils.py`

### Phase 5 (Week 5): Post-Train - **4 files**
**Priority P1**:
- `algorithms/vla_training/sft_algorithm.py`
- `runners/vla/post_train/vla_sft_runner.py`

### Phase 6 (Week 6): RL Fine-tuning - **4 files**
**Priority P1**:
- `algorithms/vla_training/rl_finetune_algorithm.py`
- `runners/vla/rl/vla_rl_runner.py`

---

## Testing Strategy

Each component requires:

1. **Unit test**: Test component in isolation
   - `tests/components/actor/test_vla_actor.py`
   - `tests/networks/vlm/test_qwen3vl.py`
   - `tests/dataset/test_lerobot_dataset.py`

2. **Config test**: Serialization/deserialization
   - All configs must pass `.to_dict()` / `.from_dict()`

3. **Integration test**: End-to-end pipeline
   - `tests/runners/vla/test_vla_pretrain_runner.py`

---

## Next Steps

1. **User Decision**: Choose implementation path
   - Option 1: Start Phase 1 (Data + VLM)
   - Option 2: Create all file stubs with TODOs
   - Option 3: Further architecture refinement

2. **After Decision**: Begin implementation following priority order (P0 → P1)

3. **Validation**: Run unit tests after each component
