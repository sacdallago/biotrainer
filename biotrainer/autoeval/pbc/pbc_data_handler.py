from tqdm import tqdm
from typing import List
from pathlib import Path

from .pbc_datasets import PBC_DATASETS

from ..data_handler import AutoEvalDataHandler, AutoEvalTask


class PBCDataHandler(AutoEvalDataHandler):
    """Handles PBC dataset related operations"""

    @staticmethod
    def get_framework_name() -> str:
        return "PBC"

    @staticmethod
    def get_download_urls():
        return ["https://nextcloud.cit.tum.de/index.php/s/gLGarZgmBEDPFJE/download"]

    @staticmethod
    def _get_all_dataset_and_split_names():
        dataset_and_split_names = []
        for dataset, dataset_info in PBC_DATASETS.items():
            subsplits = dataset_info.get("subsplits", None)
            subsplits = [(dataset, subsplit) for subsplit in subsplits] if subsplits else [(dataset, None)]
            dataset_and_split_names.extend(subsplits)
        return dataset_and_split_names

    def preprocess(self, base_path: Path, min_seq_length: int, max_seq_length: int) -> None:
        """ Filters all dataset splits for sequences that fulfill the length requirements """
        for dataset, split_name in tqdm(self._get_all_dataset_and_split_names(), desc="Preprocessing datasets"):
            dataset_dir = base_path / "supervised" / dataset
            split_file_name = dataset + "_" + split_name if split_name else dataset
            self._ensure_preprocessed_file(dataset_dir=dataset_dir,
                                           name=split_file_name,
                                           min_seq_length=min_seq_length,
                                           max_seq_length=max_seq_length)

        print("PBC data preprocessing completed!")

    def get_tasks(self, base_path: Path, min_seq_length: int, max_seq_length: int) -> List[AutoEvalTask]:
        """Build tasks for all PBC datasets"""
        tasks = []

        for dataset, split_name in self._get_all_dataset_and_split_names():
            dataset_dir = base_path / "supervised" / dataset
            split_file_name = dataset + "_" + split_name if split_name else dataset

            input_file = self._get_input_file_path(dataset_dir=dataset_dir,
                                                   name=split_file_name,
                                                   min_seq_length=min_seq_length,
                                                   max_seq_length=max_seq_length)
            if not input_file.exists():
                raise FileNotFoundError(f"Missing sequence file for {split_file_name}!")

            tasks.append(
                AutoEvalTask(framework_name=self.get_framework_name(),
                             dataset_name=dataset,
                             split_name=split_name,
                             input_file=input_file,
                             type="Protein"))

        return tasks
