"""Train the canonical RoboTwin VLA-RL task.

Usage:
    python scripts/vla/rl/train_robotwin.py \
        --task RoboTwin-PlaceCup-GRPO-v0 --num_envs 4 --max_iterations 10

The script:
  1. Initializes engines that need to load before torch (sapien/Vulkan).
  2. Imports ``RRF_robotwin_tasks.vla_rl`` to register the RoboTwin task ID.
  3. Hands off to the shared ``_common.run`` helper.
"""

from __future__ import annotations

import os
import sys

# Ensure scripts/vla/rl/_common.py is importable when running as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 1. Engine init — sapien must import before torch on some builds. Optional;
#    skipped silently if RoboTwin sim deps are not installed (the actual
#    env build will then surface a clear error).
try:
    import sapien  # noqa: F401
except ImportError:
    pass

import torch  # noqa: F401

# 2. Register all RoboTwin VLA-RL task IDs
import RRF_robotwin_tasks.vla_rl  # noqa: F401

from _common import make_parser, run


def main() -> int:
    parser = make_parser("RoboTwin")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
