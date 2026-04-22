"""
RRF D4RL Tasks

D4RL offline RL benchmark integration for RoboRenForce.
Provides state-based environments for offline RL algorithm verification.

Tasks: walker2d, hopper, halfcheetah (medium, medium-replay, medium-expert, expert)

Provides:
- VecEnv wrapper (directly uses upstream d4rl + gym packages, state-only)
- Gymnasium-registered tasks for offline RL training (IQL, CQL, etc.)
"""
