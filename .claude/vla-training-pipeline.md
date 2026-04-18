# VLA Training Pipeline Architecture

**Three-Stage Training Framework**: Pretrain → Post-Train → RL Fine-tune

---

## 🎯 Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        VLA TRAINING PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

Stage 1: PRETRAIN                    (runners/vla/pretrain/)
─────────────────────────────────────────────────────────────────
📊 Data: Large-scale offline datasets (LeRobot format)
   - EgoDex (~100k trajectories)
   - HE_Raw (real robot data)
   - Custom datasets

🎓 Method: Supervised Learning
   - Action prediction (L1/L2 loss)
   - Diffusion modeling
   - Language modeling (optional)

🎯 Goal: General-purpose VLA with broad skills
   
💾 Output: pretrained_vla.pth
   - VLM backbone (Qwen3-VL-2B)
   - Action head (Regression/Diffusion)
   - Trained on diverse tasks

         ↓

Stage 2: POST-TRAIN                  (runners/vla/post_train/)
─────────────────────────────────────────────────────────────────
📊 Data: Task-specific demonstrations or preferences
   - High-quality expert demos
   - Preference pairs (DPO)
   - Domain-specific data

🎓 Method: 
   Option A - SFT (Supervised Fine-Tuning)
     - Continue supervised training on task data
     - Add KL penalty to preserve pretrained knowledge
     - Use LoRA for parameter efficiency
   
   Option B - DPO (Direct Preference Optimization)
     - Learn from preference comparisons
     - "Good trajectory vs. bad trajectory"
     - No reward model needed

🎯 Goal: Task-specialized VLA
   - Better at specific scenarios
   - Preserves general capabilities
   
💾 Output: post_trained_vla.pth
   - Fine-tuned VLM (LoRA adapters)
   - Fine-tuned action head

         ↓

Stage 3: RL FINE-TUNE                (runners/vla/rl/)
─────────────────────────────────────────────────────────────────
📊 Data: Online interaction with environment
   - Isaac Lab simulation
   - Real robot deployment

🎓 Method: Reinforcement Learning
   - PPO (on-policy)
   - SAC (off-policy)
   - Reward signal from task success
   - KL penalty to pretrained/post-trained policy

🎯 Goal: Optimal policy for specific task
   - Maximizes task reward
   - Handles distributional shift
   - Adapts to environment dynamics
   
💾 Output: rl_finetuned_vla.pth
   - RL-optimized action head
   - Frozen VLM backbone (typically)
   - LoRA adapters on action head
```

---

## 📁 File Structure

```
source/RoboRenForce/RoboRenForce/runners/vla/
│
├── __init__.py                           # Export all VLA runners
│   from .pretrain import VLAPretrainRunner, VLAPretrainRunnerCfg
│   from .post_train import VLASFTRunner, VLADPORunner
│   from .rl import VLARLOnPolicyRunner, VLARLOffPolicyRunner
│
├── pretrain/                             # Stage 1: Supervised Pretraining
│   ├── __init__.py
│   └── pretrain_runner.py
│       ├── VLAPretrainRunnerCfg(BaseRunnerCfg)
│       └── VLAPretrainRunner(BaseRunner)
│           - Dataset: LeRobot / MixedDataset
│           - Algorithm: VLAPretrainAlgorithm
│           - Loss: Action prediction (L1/L2/Diffusion)
│           - learn(num_epochs)
│
├── post_train/                           # Stage 2: Post-Training
│   ├── __init__.py
│   ├── post_train_runner.py              # Base class
│   │   ├── VLAPostTrainRunnerCfg(BaseRunnerCfg)
│   │   └── VLAPostTrainRunner(BaseRunner)
│   │       - Loads pretrained VLA
│   │       - Applies LoRA
│   │       - Continues training
│   │
│   ├── sft_runner.py                     # Supervised Fine-Tuning
│   │   ├── VLASFTRunnerCfg(VLAPostTrainRunnerCfg)
│   │   └── VLASFTRunner(VLAPostTrainRunner)
│   │       - Task-specific demonstrations
│   │       - KL regularization to pretrained
│   │       - learn(num_epochs)
│   │
│   └── dpo_runner.py                     # Direct Preference Optimization
│       ├── VLADPORunnerCfg(VLAPostTrainRunnerCfg)
│       └── VLADPORunner(VLAPostTrainRunner)
│           - Preference pairs
│           - Bradley-Terry objective
│           - learn(num_epochs)
│
└── rl/                                   # Stage 3: RL Fine-Tuning
    ├── __init__.py
    ├── rl_on_policy_runner.py            # PPO/TRPO fine-tuning
    │   ├── VLARLOnPolicyRunnerCfg(OnPolicyRunnerCfg)
    │   └── VLARLOnPolicyRunner(OnPolicyRunner)
    │       - Loads pretrained/post-trained VLA
    │       - RL with action chunking
    │       - KL penalty to prevent forgetting
    │       - learn(num_iterations)
    │
    └── rl_off_policy_runner.py           # SAC/TD3 fine-tuning
        ├── VLARLOffPolicyRunnerCfg(OffPolicyRunnerCfg)
        └── VLARLOffPolicyRunner(OffPolicyRunner)
            - Off-policy RL
            - Multimodal replay buffer
            - learn(num_iterations)
```

---

## 🔄 Training Workflow Examples

### Example 1: Full Pipeline (Pretrain → SFT → PPO)

```python
# ===== Stage 1: Pretrain =====
from demo_tasks.vla_pretrain import HumanoidQwen3VLPretrainCfg

pretrain_cfg = HumanoidQwen3VLPretrainCfg()
pretrain_runner = pretrain_cfg.construct_from_cfg(log_dir="logs/pretrain")
pretrain_runner.learn(num_epochs=100)
# Output: checkpoints/pretrained_vla.pth

# ===== Stage 2: SFT (Post-Train) =====
from RoboRenForce.runners.vla import VLASFTRunnerCfg

sft_cfg = VLASFTRunnerCfg(
    pretrained_vla_path="checkpoints/pretrained_vla.pth",
    dataset_cfg=LeRobotDatasetCfg(
        data_root="data/humanoid_reach_expert"  # Task-specific demos
    ),
    freeze_vlm_backbone=True,
    lora_rank=8,
    regularization_weight=0.1,  # KL to pretrained
)
sft_runner = sft_cfg.construct_from_cfg(log_dir="logs/sft")
sft_runner.learn(num_epochs=20)
# Output: checkpoints/sft_vla.pth

# ===== Stage 3: RL Fine-tune =====
from demo_tasks.vla_finetune import HumanoidPPOFinetuneCfg

rl_cfg = HumanoidPPOFinetuneCfg(
    pretrained_vla_path="checkpoints/sft_vla.pth",  # Load SFT model
    kl_penalty_coef=0.01,  # KL to SFT policy
)
rl_runner = rl_cfg.construct_from_cfg(env, log_dir="logs/rl_finetune")
rl_runner.learn(num_iterations=10_000)
# Output: checkpoints/rl_finetuned_vla.pth
```

### Example 2: Skip Post-Train (Pretrain → RL directly)

```python
# Stage 1: Pretrain
pretrain_runner.learn(num_epochs=100)

# Stage 3: RL (skip SFT)
rl_cfg = HumanoidPPOFinetuneCfg(
    pretrained_vla_path="checkpoints/pretrained_vla.pth",  # Directly from pretrain
    kl_penalty_coef=0.05,  # Higher KL penalty since no SFT
)
rl_runner.learn(num_iterations=20_000)  # More iterations needed
```

### Example 3: Only Pretrain (No Fine-tuning)

```python
# Stage 1: Pretrain
pretrain_runner.learn(num_epochs=100)

# Evaluate zero-shot
from scripts.renforce import evaluate_vla
evaluate_vla(
    vla_path="checkpoints/pretrained_vla.pth",
    task="Isaac-Humanoid-Reach-v0",
    num_episodes=100
)
```

---

## 🔍 When to Use Each Stage

### Pretrain (Always Required)
**Use when**:
- Starting from scratch
- Need a general-purpose VLA
- Have large-scale diverse datasets

**Skip when**:
- Already have a pretrained VLA (e.g., OpenVLA)

### Post-Train (Optional but Recommended)
**Use SFT when**:
- Have high-quality task-specific demonstrations
- Want to improve task performance without RL
- Need to adapt to new domain (sim2real)

**Use DPO when**:
- Have preference data (human feedback)
- Want to align VLA to human preferences
- Iterative improvement from user feedback

**Skip when**:
- Pretrained VLA already performs well on task
- Want to go directly to RL
- Limited task-specific data

### RL Fine-tune (Optional for Task Optimization)
**Use when**:
- Have a reward function
- Need to optimize for specific metrics
- Want to handle distributional shift
- Pretrained/SFT VLA is good but not optimal

**Skip when**:
- Pretrained/SFT VLA already solves task
- No clear reward function
- Limited computational resources

---

## 📊 Comparison Table

| Stage      | Data Type           | Training Signal   | Compute Cost | Output Quality      |
|------------|---------------------|-------------------|--------------|---------------------|
| Pretrain   | Offline datasets    | Supervised (MSE)  | High (GPUs)  | General-purpose VLA |
| SFT        | Expert demos        | Supervised (MSE)  | Medium       | Task-specialized    |
| DPO        | Preference pairs    | Preference        | Medium       | Human-aligned       |
| RL         | Online interaction  | Reward signal     | Very High    | Task-optimal        |

---

## 🎛️ Hyperparameter Guidelines

### Pretrain
```python
VLAPretrainRunnerCfg(
    batch_size=32,              # Larger = more stable
    num_epochs=100,             # Until validation loss plateaus
    learning_rate=3e-5,         # Lower for large VLMs
    warmup_steps=1000,          # 10% of total steps
    validation_freq=5,          # Check every 5 epochs
)
```

### SFT (Post-Train)
```python
VLASFTRunnerCfg(
    pretrained_vla_path="...",
    freeze_vlm_backbone=True,   # Usually freeze VLM
    lora_rank=8,                # 8-64 typical
    learning_rate=1e-4,         # Higher than pretrain
    regularization_weight=0.1,  # KL penalty strength
    num_epochs=20,              # Fewer epochs than pretrain
)
```

### RL Fine-tune (On-Policy)
```python
VLARLOnPolicyRunnerCfg(
    pretrained_vla_path="...",
    freeze_vlm_backbone=True,   # Freeze to prevent forgetting
    rl_lora_rank=8,             # LoRA on action head
    kl_penalty_coef=0.01,       # 0.01-0.1 typical
    learning_rate=3e-5,         # Lower than PPO baseline
    num_iterations=10_000,      # Until task success rate saturates
)
```

---

## 🔬 Monitoring & Debugging

### Pretrain Metrics
```python
# Training
- action_loss: L1/L2 error (target: <0.1)
- grad_norm: Gradient norm (watch for explosions)
- learning_rate: Should decay over time

# Validation
- val_action_mse: MSE on validation set
- val_prediction_accuracy: % within threshold
```

### SFT Metrics
```python
# Training
- action_loss: Should decrease quickly
- kl_divergence: KL to pretrained (target: <0.5)
- regularization_loss: Total loss with KL penalty

# Validation
- task_success_rate: % episodes successful
- distribution_shift: KL to pretrained on val data
```

### RL Metrics
```python
# Training
- episode_return: Mean reward per episode
- success_rate: % episodes with task success
- policy_kl: KL to pretrained policy
- rl_loss: PPO/SAC loss

# Evaluation
- zero_shot_success: Success without RL
- rl_improvement: Success after RL - zero_shot
- sample_efficiency: Success vs. env steps
```

---

## 🚨 Common Issues & Solutions

### Issue 1: Catastrophic Forgetting in RL
**Symptom**: RL-finetuned VLA forgets pretrained skills

**Solution**:
```python
# Increase KL penalty
kl_penalty_coef=0.05  # Default 0.01

# Add pretrain auxiliary loss
pretrain_aux_loss_coef=0.1

# Use smaller LoRA rank
rl_lora_rank=4  # Default 8
```

### Issue 2: SFT Overfitting
**Symptom**: Perfect training loss, poor validation

**Solution**:
```python
# Increase regularization
regularization_weight=0.2  # Default 0.1

# Early stopping
early_stopping_patience=3  # epochs

# Data augmentation
image_augmentation=True
```

### Issue 3: RL Not Improving
**Symptom**: RL success rate stuck at pretrained level

**Solution**:
```python
# Reduce KL penalty (allow more exploration)
kl_penalty_coef=0.001  # Lower

# Increase exploration
exploration_noise_std=0.2  # Higher

# Check reward function
# Make sure reward is dense, not just sparse success
```

---

## 📖 Reference Implementations

- **Pretrain**: `.references/Psi0/src/psi/trainers/pretrain.py`
- **SFT**: Standard supervised fine-tuning
- **DPO**: [Direct Preference Optimization paper](https://arxiv.org/abs/2305.18290)
- **RL**: `.references/RoboTwin` + existing RoboRenforce runners

---

## 🎓 Tutorial Quick Links

1. **[Tutorial 01: VLA Pretrain](docs/tutorials/01_vla_pretrain.md)** - Start here
2. **[Tutorial 02: SFT Post-Train](docs/tutorials/02_sft_post_train.md)** - Optional stage
3. **[Tutorial 03: RL Fine-tune](docs/tutorials/03_rl_finetune.md)** - Final optimization

---

## Summary

The three-stage pipeline provides **flexibility**:
- **Pretrain**: Foundation model with broad skills ✅
- **Post-Train**: Task adaptation without RL (optional) ✅
- **RL**: Task optimization with reward signal (optional) ✅

Each stage builds on the previous, but you can **skip stages** based on your needs and available data.
