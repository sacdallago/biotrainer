from ..core import AutoEvalFramework, AutoEvalMode
from .pbc_config_bank import PBCConfigBank
from .pbc_data_handler import PBCDataHandler


class PBCFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "PBC"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.SUPERVISED

    def get_data_handler(self):
        return PBCDataHandler()

    def get_config_bank(self):
        return PBCConfigBank()
