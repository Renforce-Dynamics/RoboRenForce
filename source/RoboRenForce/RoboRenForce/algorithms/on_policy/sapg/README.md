# SAPG: Split and Aggregate Policy Gradients

## 📋 What is SAPG?

**SAPG (Split and Aggregate Policy Gradients)** is an on-policy reinforcement learning algorithm designed to improve sample efficiency in large-scale parallel environments. It was introduced in ICML 2024 (Oral).

**Paper**: [Split and Aggregate Policy Gradients (arXiv:2407.20230)](https://arxiv.org/abs/2407.20230)

## 🎯 Goal

The primary goal of SAPG is to **effectively leverage large-scale parallel environments** beyond what traditional on-policy algorithms (e.g., PPO) can achieve. Traditional algorithms often see performance saturation when the number of parallel environments increases beyond a certain point. SAPG addresses this by:

1. **Splitting** environments into multiple blocks with different exploration strategies
2. **Aggregating** experiences from different blocks using importance sampling

This enables the algorithm to benefit from diverse exploration while maintaining the stability of on-policy learning.

## 🔬 Principle

### Core Idea

SAPG works by:

1. **Split**: Divide `N` parallel environments into `K` blocks, where each block uses a different exploration coefficient
   ```
   Block 0: [env_0, ..., env_block_size-1]      → High exploration (α = 0.5)
   Block 1: [env_block_size, ..., env_2*block_size-1] → Medium exploration (α = 0.3)
   ...
   Block K-1: [env_(K-1)*block_size, ..., env_N-1]   → Low exploration (α = 0.0)
   ```

2. **Aggregate**: Use importance sampling to combine experiences from different blocks
   ```
   Policy Gradient = E_block_i [importance_weight_i × policy_gradient_i]
   ```

### Key Components

1. **Exploration Coefficient Embedding**: Each block has a unique embedding vector that conditions the policy network
   - The embedding is concatenated to observations
   - The policy network learns to adapt based on the exploration coefficient

2. **Batch Augmentation**: 
   - Select blocks to repeat (importance sampling)
   - Modify observations with different exploration coefficients
   - Recompute values and returns with new exploration coefficients
   - Concatenate augmented data for policy update

3. **Importance Sampling**: 
   - Implicitly implemented through data repetition and value recomputation
   - Allows using experiences from different exploration strategies

## 🏗️ Implementation in RoboRenForce

### Module Structure

```
algorithms/on_policy/sapg/
├── exploration_coefficient.py  # Exploration coefficient manager
├── sapg_augmentation.py         # Batch augmentation logic
└── README.md                    # This file
```

### Key Classes

1. **`ExplorationCoefficient`**: Manages exploration coefficients for different blocks
   - Generates embeddings for each block
   - Manages reward coefficients for intrinsic rewards
   - Provides methods to augment observations

2. **`SAPGBatchAugmenter`**: Implements batch augmentation
   - Selects blocks to repeat
   - Modifies observations and recomputes values/returns
   - Supports Leader-Follower mode

3. **`SAPGOnPolicyRunner`**: Runner with SAPG support
   - Extends `OnPolicyRunner`
   - Integrates exploration coefficient management
   - Applies batch augmentation before algorithm update

## 🚀 Usage

### Basic Example

```python
from RoboRenForce.runners.on_policy import SAPGOnPolicyRunner, SAPGOnPolicyRunnerCfg
from RoboRenForce.algorithms.on_policy.sapg import ExplorationCoefficientCfg
from RoboRenForce.algorithms.on_policy.ppo import PPOCfg
from RoboRenForce.components.actor_critic_pack import ActorCriticPackCfg

# 1. Configure exploration coefficients
expl_coef_cfg = ExplorationCoefficientCfg(
    expl_type="mixed_expl_learn_param",  # Scalar embedding mode
    num_blocks=6,                        # Split into 6 blocks
    embd_size=1,                         # Embedding dimension (1 for learn_param)
    embd_init_range=(50.0, 0.0),        # Embedding initialization range
    reward_coef_type="entropy",          # Entropy-based intrinsic reward
    reward_coef_scale=0.005,             # Reward coefficient scale
    reward_coef_range=(0.5, 0.0),       # Reward coefficient range
)

# 2. Configure SAPG runner
runner_cfg = SAPGOnPolicyRunnerCfg(
    # Basic configuration
    num_steps_per_env=2048,
    max_iterations=10000,
    policy=ActorCriticPackCfg(...),      # Your actor-critic config
    algorithm=PPOCfg(...),                # Your PPO config
    
    # SAPG specific configuration
    exploration_coef_cfg=expl_coef_cfg,
    off_policy_ratio=1.0,                # Repeat 1 additional block
    use_leader_follower=False,           # Use all blocks' data
    use_batch_augmentation=True,          # Enable batch augmentation
)

# 3. Create and run
runner = runner_cfg.construct_from_cfg(
    env=env,
    log_dir="./logs/sapg_experiment",
    device="cuda"
)

runner.learn(num_learning_iterations=10000)
```

### Configuration Options

#### Exploration Coefficient Configuration

- **`expl_type`**: Exploration type
  - `"mixed_expl_learn_param"`: Scalar embedding (recommended, simpler)
  - `"mixed_expl_disjoint"`: Sinusoidal encoding embedding

- **`num_blocks`**: Number of blocks to split environments into
  - Must divide `num_envs` evenly
  - Typical values: 4, 6, 8
  - Example: If `num_envs=24576` and `num_blocks=6`, each block has 4096 environments

- **`embd_size`**: Embedding dimension
  - `1` for `learn_param` mode
  - `32` or higher for `disjoint` mode

- **`reward_coef_type`**: Intrinsic reward type
  - `"entropy"`: Entropy-based intrinsic reward (encourages exploration)
  - `"none"`: No intrinsic reward

- **`reward_coef_scale`**: Scale factor for reward coefficients
  - Typical values: 0.001 to 0.01
  - Controls the strength of intrinsic rewards

#### Runner Configuration

- **`off_policy_ratio`**: Controls how many blocks to repeat
  - `1.0`: Repeat 1 additional block (total 2 blocks in batch)
  - `2.0`: Repeat 2 additional blocks (total 3 blocks in batch)
  - Higher values increase diversity but also computation cost

- **`use_leader_follower`**: Leader-Follower mode
  - `False`: Use all blocks' data (default, more data)
  - `True`: Only use leader block (block 0) data (more efficient, less data)

- **`use_batch_augmentation`**: Enable batch augmentation
  - `True`: Enable SAPG batch augmentation (default)
  - `False`: Disable (fallback to standard PPO)

## 📊 How It Works

### Training Flow

```
1. Environment Interaction
   ├─ Split N environments into K blocks
   ├─ Each block uses different exploration coefficient embedding
   └─ Collect rollout data

2. Batch Augmentation (if enabled)
   ├─ Select blocks to repeat (importance sampling)
   ├─ For each selected block:
   │  ├─ Modify observations with block's exploration coefficient
   │  ├─ Recompute values using modified observations
   │  └─ Recompute returns with new exploration coefficient
   └─ Concatenate all blocks' data

3. Policy Update
   ├─ Use augmented batch for policy gradient computation
   └─ Update policy network
```

### Observation Space Extension

The observation space is automatically extended to include exploration coefficient embeddings:

```
Original observation: [obs_dim]
Extended observation: [obs_dim + embd_dim]
                      └─ Last embd_dim dimensions are exploration coefficient embeddings
```

**Important**: Make sure your Actor and Critic networks can handle the extended observation dimension.

## ⚙️ Requirements

1. **Environment Count**: `num_envs` must be divisible by `num_blocks`
   ```python
   assert num_envs % num_blocks == 0
   ```

2. **Network Architecture**: Actor and Critic must accept extended observation dimension
   - The framework automatically extends the observation space
   - Networks are initialized with the extended dimension

3. **Memory**: Batch augmentation increases memory usage
   - Original batch size: `horizon × num_envs`
   - Augmented batch size: `horizon × num_envs × (1 + off_policy_ratio)`

## 🎯 When to Use SAPG

**Use SAPG when**:
- ✅ You have large-scale parallel environments (>10K environments)
- ✅ Traditional PPO performance saturates with more environments
- ✅ You want to improve sample efficiency
- ✅ You have sufficient computational resources

**Consider alternatives when**:
- ❌ You have limited parallel environments (<1K)
- ❌ Memory is constrained
- ❌ You prefer simpler algorithms

## 📚 References

1. **SAPG Paper**: [Split and Aggregate Policy Gradients (ICML 2024)](https://arxiv.org/abs/2407.20230)
2. **Project Website**: [sapg-rl.github.io](https://sapg-rl.github.io/)
3. **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)