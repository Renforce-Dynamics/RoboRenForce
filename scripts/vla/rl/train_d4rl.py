"""Train any D4RL-* VLA-RL task.

NOTE: D4RL is currently a placeholder; ``build()`` raises NotImplementedError
until the offline-to-online state-only path is wired.

Example:
    python scripts/vla/rl/train_d4rl.py --task D4RL-Hopper-SAC-v0
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: F401

import RRF_d4rl_tasks.vla_rl  # noqa: F401

from _common import make_parser, run


def main() -> int:
    args = make_parser("D4RL").parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
