# EPO: Evolutionary Policy Optimization

## Overview

**EPO (Evolutionary Policy Optimization)** is an extension of SAPG (Split and Aggregate Policy Gradients) that adds block-level evolutionary mechanisms to improve exploration efficiency in large-scale parallel environments.

EPO operates on a **single policy** with multiple exploration blocks, periodically evolving block parameters based on performance metrics.

## Core Idea

EPO divides parallel environments into multiple blocks, each using different exploration coefficients (similar to SAPG). The key innovation is:

1. **Track performance per block**: Monitor objective values (e.g., `true_objective` from environment) for each block.
2. **Periodic evolution**: Every `interval_steps`, identify best and worst performing blocks.
3. **Parameter merging**: Overwrite the worst block's exploration parameters with those from the best blocks.

This creates an evolutionary pressure that gradually improves exploration strategies within a single policy, without requiring multiple separate policies or population-based training.

## Implementation

The EPO module consists of three main components:

### 1. `EPOExplorationCoefficient`

Extends `SAPGExplorationCoefficient` with `merge_block_params()` method that can:
- Copy parameters from best blocks to worst blocks
- Average parameters from multiple best blocks

### 2. `EPOObserver`

Monitors training and triggers evolution:
- Collects `true_objective` values per block from environment `infos`
- Tracks best objective per block within each evolution interval
- Calls `merge_block_params()` when evolution conditions are met

### 3. `EPOOnPolicyRunner`

Extends `SAPGOnPolicyRunner` to:
- Use `EPOExplorationCoefficient` instead of standard exploration coefficient
- Initialize and manage `EPOObserver`
- Integrate observer calls into training loop

## Usage

### Basic Configuration

```python
from RoboRenForce.runners.on_policy.massive_parallel import EPOOnPolicyRunnerCfg
from RoboRenForce.algorithms.on_policy.epo import EPOExplorationCoefficientCfg

@configclass
class MyEPOCfg(EPOOnPolicyRunnerCfg):
    # Standard SAPG configuration
    exploration_coef_cfg = EPOExplorationCoefficientCfg(
        num_blocks=4,
        expl_type="mixed_expl_learn_param",
        # ... other SAPG params
    )
    
    # EPO-specific parameters
    epo_interval_steps: int = 10_000_000  # Evolution every 10M steps
    epo_warmup_steps: int = 50_000_000    # Wait 50M steps before first evolution
```

### Requirements

- **Environment must provide `true_objective` in `infos` dict** (or EPO will not trigger)
  - `true_objective` is a **task-level performance metric** used to evaluate block performance
  - **Can be reward or a separate metric:**
    - **Option 1: Use reward** - Set `self.extras['true_objective'] = self.rew_buf` (or episode reward)
      - Simple and works for most tasks
      - Note: If using reward, EPO should not modify reward-shaping parameters (EPO only evolves exploration coefficients, so this is fine)
    - **Option 2: Use separate metric** - Set `self.extras['true_objective'] = success_rate` or similar
      - Better when you want to optimize reward shaping itself
      - Examples:
        - Manipulation tasks: success rate (0.0 or 1.0 per episode)
        - Locomotion tasks: forward velocity, distance traveled
        - General: any scalar metric that reflects task performance
  - In IsaacGym/IsaacLab environments, this is set via `self.extras['true_objective'] = ...` in the environment's step function
  - **Fallback behavior**: If `true_objective` is not provided, EPO automatically uses cumulative episode reward as the objective
  - This makes EPO work out-of-the-box even when environments don't explicitly provide `true_objective`
- Use `EPOOnPolicyRunner` (or `EPOOnPolicyRunnerCfg`) instead of `SAPGOnPolicyRunner`
- All other SAPG configurations apply (batch augmentation, leader-follower, etc.)

### Example

See `RRF_isaaclab_tasks/isaaclab/locomotion/agents_sapg.py` for a complete example with `LocoRLEPOCfgBase`.

## Key Parameters

- **`epo_interval_steps`**: How often to trigger evolution (in environment frames)
- **`epo_warmup_steps`**: Initial warmup period before evolution starts
- **`num_blocks`**: Number of exploration blocks (more blocks = more diversity, but slower convergence)

## Relationship to SAPG

EPO builds on top of SAPG:
- Uses the same block-based exploration mechanism
- Reuses batch augmentation and leader-follower modes
- Adds evolutionary selection on top of the exploration diversity
