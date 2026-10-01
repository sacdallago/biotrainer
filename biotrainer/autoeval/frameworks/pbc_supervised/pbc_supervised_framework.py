from biotrainer_core.data_classes.autoeval import AutoEvalMode, AutoEvalDevMode

from .pbc_supervised_config_bank import PBCConfigBank
from .pbc_supervised_data_handler import PBCSupervisedDataHandler

from ...core import AutoEvalFramework


class PBCSupervisedFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "PBC_SUPERVISED"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.SUPERVISED

    @staticmethod
    def get_dev_mode() -> AutoEvalDevMode:
        return AutoEvalDevMode.TRAIN_VAL_TEST

    def make_data_handler(self):
        return PBCSupervisedDataHandler()

    def make_config_bank(self):
        return PBCConfigBank()
