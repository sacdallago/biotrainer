from .pbc_config_bank import PBCConfigBank
from .pbc_data_handler import PBCDataHandler

from ...core import AutoEvalFramework, AutoEvalMode


class PBCSupervisedFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "PBC_SUPERVISED"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.SUPERVISED

    def make_data_handler(self):
        return PBCDataHandler()

    def make_config_bank(self):
        return PBCConfigBank()
