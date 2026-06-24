import os

from pathlib import Path
from typing import Optional, List, Union
from appdirs import user_cache_dir

from ..core import AutoEvalDataHandler, AutoEvalTask


class ZeroShotContactDataHandler(AutoEvalDataHandler):
    """Handles contact datasets for zeroshot"""
    @staticmethod
    def get_framework_name() -> str:
        return "ZEROSHOT_CONTACT"

    @staticmethod
    def get_download_urls():
        return ["https://nextcloud.cit.tum.de/index.php/s/Q4dmpDNkNYtHiQe/download"]

    @staticmethod
    def _get_all_dataset_dirs(base_path: Path) -> List[Path]:
        dataset_dirs = sorted([base_path / d for d in os.listdir(base_path)
                            if (base_path / d).is_dir()])
        #TODO: verify folder paths!
        for dataset_dir in dataset_dirs:
            if not (dataset_dir / "extracted_sequences.fasta").exists():
                raise FileNotFoundError(f"Missing FASTA file in {dataset_dir}")
            if not (dataset_dir / "contacts").is_dir():
                raise FileNotFoundError(f"Missing contacts directory in {dataset_dir}")
        return dataset_dirs

    def preprocess(self, base_path: Path, min_seq_length: Optional[int], max_seq_length: Optional[int]) -> None:
        print("Contact datasets preprocessing completed (nothing to do)!")

    def get_tasks(self, base_path: Path, min_seq_length: Optional[int], max_seq_length: Optional[int]) -> List[
        AutoEvalTask]:
        """Build tasks for all contact datasets"""
        # fasta_file_path = dataset_dir / "extracted_sequences.fasta"
        # contacts_dir_path = dataset_dir / "contacts"
        # TODO: review choice of input files/dirs!
        return [AutoEvalTask(framework_name=self.get_framework_name(),
                             dataset_name=dataset_dir.name,
                             input_files=[dataset_dir],
                             #input_files=[fasta_file_path, contacts_dir_path],
                             type="Protein")
                for dataset_dir in self._get_all_dataset_dirs(base_path)]


#TODO: Implement SupervisedContactDataHandler here!