"""
G1 PickApple — RL Fine-tune with PPO + Regression Head

RL fine-tuning config for G1 PickApple after pretraining.
Loads pretrained VLA, applies LoRA, trains with PPO in simulation.

Usage:
    python scripts/vla/rl/train_vla_rl.py \
        --config RRF_humanoid_psi0_tasks.configs.g1_pick_apple_ppo_regression \
        --pretrained_path checkpoints/g1_pick_apple_pretrain/best.pt

TODO: Implement after VLA RL runner is ready (Phase 6).
"""


def get_config(
    pretrained_path: str = "",
    num_envs: int = 4096,
):
    """Build RL fine-tune config for G1 PickApple.

    Requires:
    - A pretrained VLA checkpoint
    - An EmbodiedEnv implementation for the G1 pick-apple task
    """
    raise NotImplementedError(
        "VLA RL fine-tuning not yet implemented. "
        "Complete Phase 6 (VLA RL Runner) first."
    )
