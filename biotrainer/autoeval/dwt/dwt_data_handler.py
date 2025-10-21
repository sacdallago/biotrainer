from tqdm import tqdm
from typing import List
from pathlib import Path

from .dwt_datasets import DWT_DATASETS

from ..data_handler import AutoEvalDataHandler, AutoEvalTask


class DWTDataHandler(AutoEvalDataHandler):
    """Handles DWT dataset related operations"""

    @staticmethod
    def get_framework_name() -> str:
        return "DWT"

    @staticmethod
    def get_download_urls():
        return ["https://nextcloud.cit.tum.de/index.php/s/DQH687EDKX6gqK6/download"]

    def preprocess(self, base_path: Path, min_seq_length: int, max_seq_length: int) -> None:
        """ Filters all dataset splits for sequences that fulfill the length requirements """
        for dataset, dataset_info in tqdm(DWT_DATASETS.items(), desc="Preprocessing datasets"):
            dataset_dir = base_path / "supervised" / dataset
            self._ensure_preprocessed_file(dataset_dir=dataset_dir,
                                           name=dataset,
                                           min_seq_length=min_seq_length,
                                           max_seq_length=max_seq_length)

        print("DWT data preprocessing completed!")

    def get_tasks(self, base_path: Path, min_seq_length: int, max_seq_length: int) -> List[AutoEvalTask]:
        """Build tasks for all DWT datasets"""
        tasks = []

        for dataset, dataset_info in DWT_DATASETS.items():
            dataset_dir = base_path / "supervised" / dataset

            input_file = self._get_input_file_path(dataset_dir=dataset_dir,
                                                   name=dataset,
                                                   min_seq_length=min_seq_length,
                                                   max_seq_length=max_seq_length)
            if not input_file.exists():
                raise FileNotFoundError(f"Missing sequence file for {dataset}!")

            tasks.append(
                AutoEvalTask(name=f"{self.get_framework_name()}-{dataset}", input_file=input_file,
                             type="Protein"))

        return tasks
