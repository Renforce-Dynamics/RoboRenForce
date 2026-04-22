"""
RRF D4RL Tasks

D4RL offline RL benchmark integration for RoboRenForce.
Provides state-based environments for offline RL algorithm verification.

Tasks: walker2d, hopper, halfcheetah (medium, medium-replay, medium-expert, expert)

Provides:
- EmbodiedEnv wrapper via RLinf's D4RLEnv (state-only, no images)
- Gymnasium-registered tasks for offline RL training (IQL, CQL, etc.)
"""
