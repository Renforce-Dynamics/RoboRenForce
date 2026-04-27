"""GR00T modality config for Psi0 (USC-PSI-Lab) G1 simple-task datasets.

Matches the schema in `<dataset>/meta/modality.json` for Psi0 sim tasks
such as `G1WholebodyTabletopGraspMP-v0`. Differs from GR00T's built-in
`unitree_g1_full_body_with_waist_height_nav_cmd` tag — Psi0 has no
left/right_leg state and uses torso velocity + target_yaw instead of
base_height/navigate_command. Video stream is `rs_view` not `ego_view`.

Use:
    bash examples/finetune.sh \\
        --dataset-path <path>/G1WholebodyTabletopGraspMP-v0 \\
        --modality-config-path examples/psi0_g1/psi0_g1_simple_config.py \\
        --embodiment-tag new_embodiment ...
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# State (32D total) — split per Psi0 modality.json
PSI0_G1_STATE_KEYS = ["left_hand", "right_hand", "left_arm", "right_arm", "rpy", "height"]

# Action (36D total) — Psi0 raw action layout
PSI0_G1_ACTION_KEYS = [
    "left_hand", "right_hand", "left_arm", "right_arm",
    "rpy", "height",
    "torso_vx", "torso_vy", "torso_vyaw", "target_yaw",
]

# Hands are binary-ish gripper signals → ABSOLUTE; arms are joint deltas → RELATIVE.
# Velocity commands are inherently RELATIVE (delta over dt).
PSI0_G1_ACTION_CONFIGS = [
    ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # left_hand
    ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # right_hand
    ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # left_arm
    ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # right_arm
    ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # rpy
    ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # height
    ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # torso_vx
    ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # torso_vy
    ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # torso_vyaw
    ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # target_yaw
]

psi0_g1_simple_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["rs_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=PSI0_G1_STATE_KEYS,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=PSI0_G1_ACTION_KEYS,
        action_configs=PSI0_G1_ACTION_CONFIGS,
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(psi0_g1_simple_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
