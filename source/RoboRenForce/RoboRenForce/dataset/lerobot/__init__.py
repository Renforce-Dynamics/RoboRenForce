# Backward compatibility — canonical location is now prototype.embodied.lerobot
from RoboRenForce.prototype.embodied.lerobot.lerobot_dataset import LeRobotDataset, LeRobotDatasetCfg

try:
    from RoboRenForce.prototype.embodied.lerobot.lerobot_processor import LeRobotProcessor, LeRobotProcessorCfg
except ImportError:
    pass
