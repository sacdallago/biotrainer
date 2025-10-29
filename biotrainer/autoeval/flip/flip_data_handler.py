from tqdm import tqdm
from typing import List
from pathlib import Path

from .flip_datasets import FLIP_DATASETS

from ..data_handler import AutoEvalDataHandler, AutoEvalTask


class FLIPDataHandler(AutoEvalDataHandler):
    """Handles FLIP dataset related operations"""

    IGNORE_SPLITS = ["mixed_vs_human_2"]

    @staticmethod
    def get_framework_name() -> str:
        return "FLIP"

    @staticmethod
    def get_download_urls():
        return ["https://nextcloud.cit.tum.de/index.php/s/fqFEeCSpwTHkt8X/download"]

    def preprocess(self, base_path: Path, min_seq_length: int, max_seq_length: int) -> None:
        """ Filters all dataset splits for sequences that fulfill the length requirements """
        for dataset, dataset_info in tqdm(FLIP_DATASETS.items(), desc="Preprocessing datasets"):
            dataset_dir = base_path / dataset

            # Process all splits
            for split in dataset_info["splits"]:
                if split in self.IGNORE_SPLITS:
                    continue

                self._ensure_preprocessed_file(dataset_dir=dataset_dir,
                                               name=split,
                                               min_seq_length=min_seq_length,
                                               max_seq_length=max_seq_length)

        print("FLIP data preprocessing completed!")

    def get_tasks(self, base_path: Path, min_seq_length: int, max_seq_length: int) -> List[AutoEvalTask]:
        """Build tasks for all FLIP datasets"""
        print("WARNING: FLIP dataset support is currently deprecated in biotrainer - please refer to the PBC datasets "
              "instead!")

        tasks = []

        for dataset, dataset_info in FLIP_DATASETS.items():
            dataset_dir = base_path / dataset
            for split_name in dataset_info["splits"]:
                if split_name in self.IGNORE_SPLITS:
                    continue

                input_file = self._get_input_file_path(dataset_dir=dataset_dir,
                                                       name=split_name,
                                                       min_seq_length=min_seq_length,
                                                       max_seq_length=max_seq_length)
                if not input_file.exists():
                    raise FileNotFoundError(f"Missing sequence file for {dataset}/{split_name}!")

                tasks.append(
                    AutoEvalTask(name=f"{self.get_framework_name()}-{dataset}-{split_name}", input_file=input_file,
                                 type="Protein"))

        return tasks
