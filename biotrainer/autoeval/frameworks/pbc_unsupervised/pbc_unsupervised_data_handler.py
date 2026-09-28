from typing import List, Tuple, Optional

from ..pbc_supervised.pbc_supervised_data_handler import PBCSupervisedDataHandler


class PBCUnsupervisedDataHandler(PBCSupervisedDataHandler):
    """
    Re-uses supervised logic for data handler
    """

    def _get_all_dataset_and_split_names(self) -> List[Tuple[str, Optional[str]]]:
        return [("cath", None)]

    @staticmethod
    def get_framework_name():
        return "PBC_UNSUPERVISED"

    def get_framework_subdirectory(self) -> str:
        return "unsupervised"  # PBC/unsupervised

    @staticmethod
    def get_download_urls():
        return ["https://nextcloud.cit.tum.de/index.php/s/N83oNAnwHznziLg/download"]
