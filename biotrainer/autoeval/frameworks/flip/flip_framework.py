from biotrainer_core.data_classes.autoeval import AutoEvalMode, AutoEvalDevMode

from .flip_config_bank import FLIPConfigBank
from .flip_data_handler import FLIPDataHandler

from ...core import AutoEvalFramework


class FLIPFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "FLIP"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.SUPERVISED

    @staticmethod
    def get_dev_mode() -> AutoEvalDevMode:
        return AutoEvalDevMode.TRAIN_VAL_TEST

    def make_data_handler(self):
        return FLIPDataHandler()

    def make_config_bank(self):
        return FLIPConfigBank()
