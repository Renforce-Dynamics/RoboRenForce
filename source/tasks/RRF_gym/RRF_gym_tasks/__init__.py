"""
RRF Gym Tasks

Classic Gymnasium environments (MuJoCo locomotion, classic control)
wrapped as RoboRenForce tasks with registered env IDs.

Registered IDs follow the pattern: RoboRenForce-Gym-{EnvName}-{Algorithm}
e.g. RoboRenForce-Gym-HalfCheetah-SAC, RoboRenForce-Gym-Walker2d-PPO
"""

from . import locomotion  # triggers registration
