"""RRF_orchestra — process-level parallelism for VLA RL training.

See docs/PLAN-task3-orchestra-package.md for the design.
"""

__version__ = "0.1.0"

from RRF_orchestra.protocol.messages import PROTOCOL_VERSION

__all__ = ["__version__", "PROTOCOL_VERSION"]
