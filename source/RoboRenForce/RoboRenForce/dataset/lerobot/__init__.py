# Backward compatibility — canonical location is now prototype.embodied_ai.lerobot
from RoboRenForce.prototype.embodied_ai.lerobot.lerobot_dataset import LeRobotDataset, LeRobotDatasetCfg

try:
    from RoboRenForce.prototype.embodied_ai.lerobot.lerobot_processor import LeRobotProcessor, LeRobotProcessorCfg
except ImportError:
    pass
