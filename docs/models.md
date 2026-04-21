# RoboRenForce — Supported VLA Models

## Quick Start

```bash
# List available models
python -c "from RRF_models.registry import list_models; print(list_models())"

# Setup a model (downloads weights + dependencies)
bash scripts/models/setup_models.sh qwen2vl

# Use in code
from RRF_models.registry import get_model
policy = get_model("qwen2vl", cfg=my_config)
```

## Model Overview

| Model | Params | Output Dim | HuggingFace ID | License |
|-------|--------|-----------|----------------|---------|
| Qwen2-VL | 2B/7B | 1536 | `Qwen/Qwen2-VL-2B-Instruct` | Apache 2.0 |
| Qwen3-VL | 2B/8B | 2048 | `Qwen/Qwen3-VL-2B-Instruct` | Apache 2.0 |
| OpenPI (pi0.5) | 4B | 2048 | `lerobot/pi05_base` | Apache 2.0 + Gemma |
| GR00T N1.7 | 3B | 2048 | `nvidia/GR00T-N1.7-3B` | Apache 2.0 |
| MLP Baseline | ~1M | 64 | — (built-in) | — |

---

## 1. Qwen2-VL

**Architecture**: Vision Transformer + Qwen2 LLM with dynamic resolution support.

### Install

```bash
pip install "transformers>=4.37" qwen-vl-utils accelerate
# Or use the setup script:
bash scripts/models/setup_models.sh qwen2vl
```

### Available Variants

| Model ID | Params | VRAM (bf16) |
|----------|--------|------------|
| `Qwen/Qwen2-VL-2B-Instruct` | 2.2B | ~5GB |
| `Qwen/Qwen2-VL-7B-Instruct` | 7.6B | ~16GB |

### Usage

```python
from RoboRenForce.networks.vlm.qwen2vl import Qwen2VLCfg
from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicy, Qwen2VLPolicyCfg
from RoboRenForce.components.actor.vla_actor import VLAActorCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import RegressionActionHeadCfg

cfg = Qwen2VLPolicyCfg(
    actor_cfg=VLAActorCfg(
        vlm_backbone_cfg=Qwen2VLCfg(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            freeze=True,           # Freeze VLM (recommended for RL)
            use_bf16=True,
        ),
        freeze_vlm=True,
        fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
        action_head_cfg=RegressionActionHeadCfg(action_dim=14, action_horizon=1),
        use_proprioception=True,
    ),
    proprio_dim=14,
    use_value_head=True,  # Enable for PPO/GRPO
)
policy = Qwen2VLPolicy(cfg)
```

---

## 2. Qwen3-VL

**Architecture**: Upgraded Qwen2-VL with larger hidden dim (2048 vs 1536) and Qwen3 language model.

### Install

```bash
pip install "transformers>=4.51" qwen-vl-utils accelerate
bash scripts/models/setup_models.sh qwen3vl
```

### Available Variants

| Model ID | Params | VRAM (bf16) |
|----------|--------|------------|
| `Qwen/Qwen3-VL-2B-Instruct` | 2.5B | ~6GB |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | ~17GB |

### Usage

```python
from RoboRenForce.networks.vlm.qwen3vl import Qwen3VLCfg
from RRF_models.qwen3vl.qwen3vl_policy import Qwen3VLPolicy, Qwen3VLPolicyCfg

cfg = Qwen3VLPolicyCfg(
    actor_cfg=VLAActorCfg(
        vlm_backbone_cfg=Qwen3VLCfg(
            model_name="Qwen/Qwen3-VL-2B-Instruct",
            freeze=True,
        ),
        freeze_vlm=True,
        fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
        action_head_cfg=RegressionActionHeadCfg(action_dim=14, action_horizon=1),
        use_proprioception=True,
    ),
    proprio_dim=14,
)
policy = Qwen3VLPolicy(cfg)
```

---

## 3. OpenPI (pi0 / pi0.5)

**Architecture**: PaliGemma/Gemma VLM + flow-matching action head. Physical Intelligence's foundation model for robot manipulation.

### Install

**Option A — Via LeRobot (recommended)**:
```bash
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
```

**Option B — Native OpenPI**:
```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git third_party/openpi
cd third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### Available Variants

| Model ID | Type | Params | Notes |
|----------|------|--------|-------|
| `lerobot/pi05_base` | HF | 4B | Recommended default |
| `lerobot/pi0_old` | HF | ~3B | Original pi0 |
| `pi0_base` | GCS | ~3B | Native OpenPI |
| `pi05_base` | GCS | 4B | Native OpenPI |
| `pi0_fast_base` | GCS | ~3B | FAST tokenization |

### Usage

```python
from RoboRenForce.networks.vlm.openpi import OpenPICfg
from RRF_models.openpi.openpi_policy import OpenPIPolicy, OpenPIPolicyCfg

# Via LeRobot (HuggingFace)
cfg = OpenPIPolicyCfg(
    actor_cfg=VLAActorCfg(
        vlm_backbone_cfg=OpenPICfg(
            model_name="lerobot/pi05_base",
            use_lerobot=True,
            freeze=True,
        ),
        freeze_vlm=True,
        fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
        action_head_cfg=RegressionActionHeadCfg(action_dim=14, action_horizon=1),
        use_proprioception=True,
    ),
    proprio_dim=14,
)
policy = OpenPIPolicy(cfg)

# Native OpenPI
cfg = OpenPIPolicyCfg(
    actor_cfg=VLAActorCfg(
        vlm_backbone_cfg=OpenPICfg(
            use_lerobot=False,
            native_config="pi05_base",
            native_checkpoint="path/to/checkpoint",
        ),
        ...
    ),
)
```

### Key Concepts

- **Flow Matching**: pi0 generates actions via iterative denoising (like diffusion but with ODE-based flow)
- **Multi-Embodiment**: Pre-trained on diverse robots. Fine-tune configs available for ALOHA, DROID, LIBERO
- **Feature Extraction**: In RoboRenForce, we extract VL features before pi0's action head and feed them into our own action pipeline. This enables using RoboRenForce's RL algorithms (GRPO, PPO, etc.) for fine-tuning.

---

## 4. GR00T N1.7

**Architecture**: SigLip2 vision + Cosmos-Reason2-2B language + flow-matching DiT action decoder.

### Install

**Option A — HuggingFace (weights only)**:
```bash
pip install "transformers>=4.40" accelerate safetensors
pip install flash-attn  # Requires CUDA
```

**Option B — Full Isaac-GR00T**:
```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T third_party/isaac-gr00t
cd third_party/isaac-gr00t
uv sync --python 3.10
```

### Available Variants

| Model ID | Notes | VRAM (bf16) |
|----------|-------|------------|
| `nvidia/GR00T-N1.7-3B` | Base model | ~7GB |
| `nvidia/GR00T-N1.7-DROID` | DROID fine-tuned | ~7GB |
| `nvidia/GR00T-N1.7-LIBERO` | LIBERO fine-tuned | ~7GB |
| `nvidia/GR00T-N1.7-SimplerEnv-Bridge` | SimplerEnv WidowX | ~7GB |

### Usage

```python
from RoboRenForce.networks.vlm.gr00t import GR00TCfg
from RRF_models.gr00t.gr00t_policy import GR00TPolicy, GR00TPolicyCfg

cfg = GR00TPolicyCfg(
    actor_cfg=VLAActorCfg(
        vlm_backbone_cfg=GR00TCfg(
            model_name="nvidia/GR00T-N1.7-3B",
            freeze=True,
            embodiment_tag="new_embodiment",
        ),
        freeze_vlm=True,
        fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
        action_head_cfg=RegressionActionHeadCfg(action_dim=14, action_horizon=1),
        use_proprioception=True,
    ),
    proprio_dim=14,
    embodiment_tag="new_embodiment",
)
policy = GR00TPolicy(cfg)
```

### Embodiment Tags

GR00T uses embodiment tags to map between proprioception/action formats:

| Tag | Robot |
|-----|-------|
| `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` | DROID (7DOF + gripper) |
| `LIBERO_PANDA` | Franka Panda in LIBERO |
| `SIMPLER_ENV_WIDOWX` | WidowX 250 |
| `new_embodiment` | Custom (define your own) |

---

## 5. MLP Baseline

Built-in lightweight baseline for testing and debugging. No external model needed.

```python
from RRF_models.registry import get_model
from RRF_models.mlp_baseline import MLPBaselinePolicyCfg

cfg = MLPBaselinePolicyCfg(
    obs_dim=64 + 14,  # image_features + proprio
    action_dim=14,
    hidden_dims=[256, 256],
)
policy = get_model("mlp_baseline", cfg=cfg)
```

---

## Architecture Overview

All models follow the same pattern in RoboRenForce:

```
EmbodiedEnv Observation
    │
    ├─ main_images [B, H, W, C]
    ├─ states [B, state_dim]
    └─ task_descriptions [str]
         │
         ▼
    Policy._map_obs()
         │
    ┌────┴─────────────────────┐
    │  VLMBackbone             │  ← Qwen2-VL / Qwen3-VL / OpenPI / GR00T
    │  image + text → features │     (frozen or fine-tuned)
    └────┬─────────────────────┘
         │ vl_features [B, output_dim]
         ▼
    ┌────┴─────────────────────┐
    │  FusionLayer             │  ← concat_mlp or add
    │  features + proprio      │
    └────┬─────────────────────┘
         │ fused [B, fusion_dim]
         ▼
    ┌────┴─────────────────────┐
    │  ActionHead              │  ← Regression MLP or Diffusion
    │  features → actions      │
    └────┬─────────────────────┘
         │ actions [B, horizon, action_dim]
         ▼
    ForwardType dispatch:
      INFERENCE → actions only
      PRETRAIN  → L1 loss vs target
      PPO/GRPO  → logprobs + values
      SFT       → action loss + KL
      SAC       → Q-values
```

## Adding a New Model

1. Create VLM backbone: `source/RoboRenForce/RoboRenForce/networks/vlm/mymodel.py`
   - Subclass `VLMBackbone`, implement `forward(image, text) → (B, output_dim)`
   - Create `MyModelCfg(VLMBackboneCfg)` with model-specific params

2. Create policy adapter: `source/RRF_models/RRF_models/mymodel/`
   - `__init__.py`: builder function + `register_model("mymodel", builder)`
   - `mymodel_policy.py`: `MyModelPolicy(BasePolicy)` with `_map_obs()`, `forward()`

3. Register: add `"mymodel": "RRF_models.mymodel"` to `_KNOWN_MODELS` in `registry.py`

4. Test: `python -c "from RRF_models.registry import list_models; print(list_models())"`
