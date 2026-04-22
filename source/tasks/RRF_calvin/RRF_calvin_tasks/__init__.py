"""
RRF CALVIN Tasks

CALVIN benchmark integration for RoboRenForce.
Long-horizon manipulation with 5-subtask sequential evaluation.

Scenes: A, B, C, D
Task suites: calvin_d, calvin_abc, calvin_abcd

Provides:
- EmbodiedEnv wrapper (directly uses upstream calvin_env package)
- Gymnasium-registered tasks for VLA training
"""
