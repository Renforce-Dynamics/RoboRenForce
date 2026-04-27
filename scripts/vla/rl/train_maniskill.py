"""Train any ManiSkill-* VLA-RL task.

Example:
    python scripts/vla/rl/train_maniskill.py --task ManiSkill-PickCube-GRPO-v0 \
        --num_envs 4 --max_iterations 10
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: F401

import RRF_maniskill_tasks.vla_rl  # noqa: F401

from _common import make_parser, run


def main() -> int:
    args = make_parser("ManiSkill").parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
