"""
Config: place_empty_cup — Supervised Pretraining with Qwen2-VL

A reference config for pretraining a VLA on RoboTwin's place_empty_cup task.
Uses offline demonstrations in LeRobot v2 format.

Usage:
    python -m RoboRenForce.runners.vla.pretrain.run --config \\
        source/tasks/RRF_robotwin/RRF_robotwin_tasks/configs/place_empty_cup_pretrain.py
"""


def get_config():
    from RoboRenForce.runners.vla.pretrain.vla_pretrain_runner import VLAPretrainRunnerCfg
    from RoboRenForce.algorithms.vla_training.pretrain_algorithm import VLAPretrainAlgorithmCfg
    from RRF_robotwin_tasks.robots.piper import PiperCfg
    from RRF_robotwin_tasks.datasets.robotwin_demo import RoboTwinDemoCfg

    robot = PiperCfg()

    dataset = RoboTwinDemoCfg(
        task_name="place_empty_cup",
        data_root="/data/robotwin/place_empty_cup",  # Update to actual path
        num_episodes=100,
        image_size=(224, 224),
    )

    return {
        "robot": robot,
        "dataset": dataset,
        "runner": VLAPretrainRunnerCfg(
            model_type="qwen2vl",       # via RRF_models registry
            model_cfg=None,             # TODO: fill in Qwen2VLPolicyCfg
            batch_size=16,
            num_epochs=10,
            num_workers=4,
            algorithm_cfg=VLAPretrainAlgorithmCfg(
                action_loss_type="l1",
                learning_rate=1e-4,
                warmup_steps=500,
            ),
        ),
    }
