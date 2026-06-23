from ..core import AutoEvalFramework, AutoEvalMode
from .contact_config_bank import ContactConfigBank
from .contact_data_handler import ZeroShotContactDataHandler


class ZeroShotContactFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "ZEROSHOT_CONTACT"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.ZERO_SHOT

    def get_data_handler(self):
        return ZeroShotContactDataHandler()

    def get_config_bank(self):
        return ContactConfigBank()
