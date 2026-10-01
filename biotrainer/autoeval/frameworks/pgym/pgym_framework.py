from biotrainer_core.data_classes.autoeval import AutoEvalMode, AutoEvalDevMode

from .pgym_config_bank import PGYMConfigBank
from .pgym_data_handler import PGYMDataHandler

from ...core import AutoEvalFramework


class PGYMFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "PGYM"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.ZERO_SHOT

    @staticmethod
    def get_dev_mode() -> AutoEvalDevMode:
        return AutoEvalDevMode.DEV_DATASET_TEST

    def make_data_handler(self):
        return PGYMDataHandler()

    def make_config_bank(self):
        return PGYMConfigBank()
