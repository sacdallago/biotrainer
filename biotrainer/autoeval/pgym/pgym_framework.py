from ..core import AutoEvalFramework, AutoEvalMode
from .pgym_config_bank import PGYMConfigBank
from .pgym_data_handler import PGYMDataHandler


class PGYMFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "PGYM"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.ZERO_SHOT

    def get_data_handler(self):
        return PGYMDataHandler()

    def get_config_bank(self):
        return PGYMConfigBank()
