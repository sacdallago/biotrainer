from tqdm import tqdm
from typing import List
from pathlib import Path

from .flip_datasets import FLIP_DATASETS

from ..data_handler import AutoEvalDataHandler, AutoEvalTask

from ...input_files import filter_FASTA


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
                if split in FLIPDataHandler.IGNORE_SPLITS:
                    continue

                FLIPDataHandler._ensure_preprocessed_file(dataset_dir=dataset_dir,
                                                          name=split,
                                                          min_seq_length=min_seq_length,
                                                          max_seq_length=max_seq_length)

        print("FLIP data preprocessing completed!")

    def get_tasks(self, base_path: Path, min_seq_length: int, max_seq_length: int) -> List[AutoEvalTask]:
        """Build path dictionary for all FLIP datasets"""
        tasks = []

        for dataset, dataset_info in FLIP_DATASETS.items():
            dataset_dir = base_path / dataset
            for split_name in dataset_info["splits"]:
                if split_name in FLIPDataHandler.IGNORE_SPLITS:
                    continue

                input_file = FLIPDataHandler._get_input_file_path(dataset_dir=dataset_dir,
                                                                  name=split_name,
                                                                  min_seq_length=min_seq_length,
                                                                  max_seq_length=max_seq_length)
                if not input_file.exists():
                    raise FileNotFoundError(f"Missing sequence file for {dataset}/{split_name}!")

                tasks.append(AutoEvalTask(name=f"{dataset}-{split_name}", input_file=input_file, type="Protein"))

        return tasks

    @staticmethod
    def _get_input_file_path(dataset_dir: Path, name: str, min_seq_length: int, max_seq_length: int) -> Path:
        """Get the appropriate input file path (preprocessed if available)"""
        raw_path = dataset_dir / f"{name}.fasta"
        preprocessed_dir_name = FLIPDataHandler.get_preprocessed_file_dir_name(min_seq_length=min_seq_length,
                                                                               max_seq_length=max_seq_length)
        preprocessed_path = dataset_dir / preprocessed_dir_name / f"{name}.fasta"

        if preprocessed_path.exists():
            return preprocessed_path
        return raw_path

    @staticmethod
    def _ensure_preprocessed_file(dataset_dir: Path, name: str, min_seq_length: int, max_seq_length: int) -> Path:
        """Ensure a preprocessed version of the file exists and return its path"""
        download_path = dataset_dir / f"{name}.fasta"
        preprocessed_dir_name = FLIPDataHandler.get_preprocessed_file_dir_name(min_seq_length=min_seq_length,
                                                                               max_seq_length=max_seq_length)
        preprocessed_path = dataset_dir / preprocessed_dir_name / f"{name}.fasta"

        # If preprocessed file already exists, return its path
        if preprocessed_path.exists():
            return preprocessed_path

        # If raw file doesn't exist, we can't proceed
        if not download_path.exists():
            raise FileNotFoundError(f"Required file {download_path} not available for: {name}")

        # Preprocess the file
        preprocessed_dir = dataset_dir / preprocessed_dir_name
        preprocessed_dir.mkdir(exist_ok=True)

        n_kept, n_all = filter_FASTA(input_path=download_path,
                                     output_path=preprocessed_path,
                                     filter_function=lambda seq_record:
                                     min_seq_length <= len(seq_record.seq) <= max_seq_length
                                     )

        print(f"Preprocessed (min: {min_seq_length}, max: {max_seq_length}) {download_path.name}: "
              f"kept {n_kept}/{n_all} sequences")
        return preprocessed_path
