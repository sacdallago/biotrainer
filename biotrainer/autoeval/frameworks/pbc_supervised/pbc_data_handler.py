from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
from biotrainer_core.data_classes.autoeval import AutoEvalTask, all_pbc_supervised_datasets

from ...core import AutoEvalDataHandler


class PBCDataHandler(AutoEvalDataHandler):
    """Handles PBC dataset related operations"""

    @staticmethod
    def get_framework_name() -> str:
        return "PBC_SUPERVISED"

    @staticmethod
    def get_download_urls():
        return ["https://nextcloud.cit.tum.de/index.php/s/gLGarZgmBEDPFJE/download"]

    @staticmethod
    def _get_all_dataset_and_split_names():
        dataset_and_split_names = []
        for dataset, dataset_info in all_pbc_supervised_datasets().items():
            splits = dataset_info.splits
            splits = [(dataset, subsplit) for subsplit in splits] if splits else [(dataset, None)]
            dataset_and_split_names.extend(splits)
        return dataset_and_split_names

    def _get_dataset_dir_name_and_split_file_name(self, dataset: str, split_name: str):
        if "-" in dataset:
            # Frustration - Has regression and classification splits in same directory
            vals = dataset.split("-")
            dataset_name = vals[0]
            task_type = vals[1]
            split_file_name = dataset_name + "_" + task_type
            return dataset_name, split_file_name

        split_file_name = dataset + "_" + split_name if split_name else dataset
        return dataset, split_file_name

    def preprocess(self, base_path: Path, min_seq_length: Optional[int], max_seq_length: Optional[int]) -> None:
        """ Filters all dataset splits for sequences that fulfill the length requirements """
        for dataset, split_name in tqdm(self._get_all_dataset_and_split_names(), desc="Preprocessing datasets"):
            dataset_name, split_file_name = self._get_dataset_dir_name_and_split_file_name(dataset, split_name)
            dataset_dir = base_path / "supervised" / dataset_name
            self._ensure_preprocessed_file(dataset_dir=dataset_dir,
                                           name=split_file_name,
                                           min_seq_length=min_seq_length,
                                           max_seq_length=max_seq_length)

        print("PBC data preprocessing completed!")

    def get_tasks(self, base_path: Path, min_seq_length: Optional[int], max_seq_length: Optional[int]) \
            -> List[AutoEvalTask]:
        """Build tasks for all PBC datasets"""
        tasks = []

        for dataset, split_name in self._get_all_dataset_and_split_names():
            dataset_dir = base_path / "supervised"
            dataset_name, split_file_name = self._get_dataset_dir_name_and_split_file_name(dataset, split_name)
            dataset_dir /= dataset_name

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
                             input_files=[input_file],
                             type="Protein",
                             ))

        return tasks
