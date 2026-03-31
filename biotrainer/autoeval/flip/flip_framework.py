from ..core import AutoEvalFramework, AutoEvalMode
from .flip_config_bank import FLIPConfigBank
from .flip_data_handler import FLIPDataHandler


class FLIPFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "FLIP"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.SUPERVISED

    def get_data_handler(self):
        return FLIPDataHandler()

    def get_config_bank(self):
        return FLIPConfigBank()
