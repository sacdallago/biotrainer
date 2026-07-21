from .pgym_config_bank import PGYMConfigBank
from .pgym_data_handler import PGYMDataHandler

from ...core import AutoEvalFramework, AutoEvalMode


class PGYMFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "PGYM"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.ZERO_SHOT

    def make_data_handler(self):
        return PGYMDataHandler()

    def make_config_bank(self):
        return PGYMConfigBank()
