"""
Shared test fixtures and setup.

Stubs out gymnasium which is needed by base_runner but not by VLA tests.
"""

import sys
import types

# Stub gymnasium if not installed (needed by base_runner -> lab_wrapper chain)
if "gymnasium" not in sys.modules:
    _gym = types.ModuleType("gymnasium")
    _gym.Env = type("Env", (), {})
    _gym.Space = type("Space", (), {})
    _gym_spaces = types.ModuleType("gymnasium.spaces")
    _gym_spaces.Box = type("Box", (), {})
    _gym_spaces.Dict = type("Dict", (), {})
    _gym_spaces.Discrete = type("Discrete", (), {})
    sys.modules["gymnasium"] = _gym
    sys.modules["gymnasium.spaces"] = _gym_spaces
