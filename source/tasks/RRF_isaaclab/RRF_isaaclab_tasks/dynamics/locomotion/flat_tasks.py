from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.flat_env_cfg import UnitreeGo1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_b.flat_env_cfg import AnymalBFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

task_names = [
    "UnitreeA1FlatEnvCfg",
    "UnitreeGo1FlatEnvCfg",
    "UnitreeGo2FlatEnvCfg",
    "AnymalBFlatEnvCfg",
    "AnymalCFlatEnvCfg",
    "AnymalDFlatEnvCfg",
    "H1FlatEnvCfg",
    "G1FlatEnvCfg",
]

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class DynamicObservationCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        
@configclass
class TerrainObservationCfg(ObsGroup):
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        
@configclass
class FullObservationCfg(ObsGroup):
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )
    actions = ObsTerm(func=mdp.last_action, clip=(-10, 10))
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        
@configclass
class NoisedFullObservationCfg(ObsGroup):
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )
    actions = ObsTerm(func=mdp.last_action, clip=(-10, 10))
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        
def modify_observation(target_cfg):
    has_tarrain = getattr(target_cfg.observations.policy, "height_scan", None) is not None
    target_cfg.observations.policy = NoisedFullObservationCfg()
    target_cfg.observations.critic = FullObservationCfg()
    target_cfg.observations.dynamic = DynamicObservationCfg()
    if has_tarrain:
        target_cfg.observations.terrain = TerrainObservationCfg()
    else:
        target_cfg.observations.policy.height_scan = None
        target_cfg.observations.critic.height_scan = None

def modify_observation_class(target_cfg_class):
    original_name = target_cfg_class.__name__
    original_module = target_cfg_class.__module__
    class ModifiedCfg(target_cfg_class):
        def __post_init__(self):
            if hasattr(super(), '__post_init__'):
                super().__post_init__()
            has_terrain = getattr(self.observations.policy, "height_scan", None) is not None
            self.observations.policy = NoisedFullObservationCfg()
            self.observations.critic = FullObservationCfg()
            self.observations.dynamic = DynamicObservationCfg()
            if has_terrain:
                self.observations.terrain = TerrainObservationCfg()
            else:
                self.observations.policy.height_scan = None
                self.observations.critic.height_scan = None
    ModifiedCfg.__name__ = f"{original_name}"
    ModifiedCfg.__qualname__ = f"{original_name}"
    ModifiedCfg.__module__ = original_module    
    return ModifiedCfg
