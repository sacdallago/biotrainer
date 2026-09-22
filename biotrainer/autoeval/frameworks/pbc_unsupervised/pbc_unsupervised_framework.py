from ...core import AutoEvalFramework, AutoEvalMode
from .pbc_unsupervised_config_bank import PBCUnsupervisedConfigBank
from .pbc_unsupervised_data_handler import PBCUnsupervisedDataHandler


class PBCUnsupervisedFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "PBC_UNSUPERVISED"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.UNSUPERVISED

    def make_data_handler(self):
        return PBCUnsupervisedDataHandler()

    def make_config_bank(self):
        return PBCUnsupervisedConfigBank()
