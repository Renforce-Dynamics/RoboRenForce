from typing import Any
from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg

@configclass
class ModuleDict(ModuleBaseCfg):
    module_dict: dict[str, ModuleBase] = {}  # type: ignore

    def construct_from_cfg(self, *args, **kwargs):
        return {
            name: module.construct_from_cfg(self, *args, **kwargs)
            for name, module in self.module_dict.items()
        }
