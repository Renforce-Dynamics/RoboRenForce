"""VLA actor cfg for RoboTwin ``place_empty_cup``.

Returns a callable that constructs the BasePolicy at training launch
(after CLI overrides are applied to the EnvCfg).
"""

from __future__ import annotations

from .._shared.vla_actor_presets import qwen2vl_actor


def build_place_cup_policy(use_value_head: bool = False):
    """Build the Qwen2-VL VLA policy for place_empty_cup.

    Lives in :mod:`RRF_models.qwen2vl` (the BasePolicy adapter that wraps
    the core ``VLAActor`` building blocks).

    Args:
        use_value_head: True for PPO (needs critic), False for GRPO.
    """
    # Import lazily so importing the task package doesn't pull in HF transformers.
    from RRF_models.qwen2vl import build_qwen2vl_policy
    from RRF_models.qwen2vl.qwen2vl_policy import Qwen2VLPolicyCfg

    cfg = Qwen2VLPolicyCfg(
        actor_cfg=qwen2vl_actor(action_dim=14, action_horizon=1, freeze_vlm=True),
        use_value_head=use_value_head,
        proprio_dim=14,
    )
    return build_qwen2vl_policy(cfg=cfg)
