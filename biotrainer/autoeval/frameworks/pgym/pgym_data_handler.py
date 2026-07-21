import os
import random
import pandas as pd

from pathlib import Path
from typing import List, Optional
from biotrainer_core.data_classes.autoeval import AutoEvalTask

from ...core import AutoEvalDataHandler


class PGYMDataHandler(AutoEvalDataHandler):
    """Handles PGYM dataset related operations"""
    DEVELOPMENT_MODE_SUBSAMPLE_RATIO = 0.4

    _file_paths_dev_mode: List[str] = []

    @staticmethod
    def get_framework_name() -> str:
        return "PGYM"

    @staticmethod
    def get_download_urls():
        return [
            "https://nextcloud.cit.tum.de/index.php/s/7NmBnkqys45tfPH/download",
            "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
        ]

    @staticmethod
    def get_reference_file_urls() -> List[str]:
        return [
            "https://nextcloud.cit.tum.de/index.php/s/9tsiQtn3TwpS2wF/download",
            "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv"
        ]

    @staticmethod
    def _get_all_files(base_path: Path):
        all_files = [file for file in os.listdir(base_path)
                     if os.path.isfile(base_path / file) and file.endswith(".csv")]
        return all_files

    def preprocess(self, base_path: Path, min_seq_length: Optional[int], max_seq_length: Optional[int]) -> None:
        print("PGYM data preprocessing completed (nothing to do)!")

    def _subsample_for_development_mode(self, file_paths: List[Path]) -> List[Path]:
        rng = random.Random(12)
        file_paths_sample = rng.sample(file_paths, int(len(file_paths) * self.DEVELOPMENT_MODE_SUBSAMPLE_RATIO))
        self._file_paths_dev_mode.extend([fp.stem for fp in file_paths_sample])
        return file_paths_sample

    def get_tasks(self, base_path: Path, min_seq_length: Optional[int], max_seq_length: Optional[int],
                  development_mode: bool) -> List[AutoEvalTask]:
        """Build tasks for all PGYM datasets"""
        substitutions_path = base_path / "DMS_ProteinGym_substitutions"
        reference_df = pd.read_csv(self.get_reference_file_path(base_path))
        file2taxon = {row["DMS_id"] + ".csv": row["taxon"] for _, row in reference_df.iterrows()}
        virus_taxon_files = [substitutions_path / file for file, taxon in file2taxon.items()
                             if taxon.lower() == "virus"]
        non_virus_taxon_files = [substitutions_path / file for file, taxon in file2taxon.items()
                                 if taxon.lower() != "virus"]
        assert len(virus_taxon_files) + len(non_virus_taxon_files) == len(file2taxon)

        # Always calculate the sample for storing the respective file paths
        virus_taxon_files_sample = self._subsample_for_development_mode(virus_taxon_files)
        non_virus_taxon_files_sample = self._subsample_for_development_mode(non_virus_taxon_files)

        if development_mode:
            virus_taxon_files = virus_taxon_files_sample
            non_virus_taxon_files = non_virus_taxon_files_sample

        for dataset in self._get_all_files(base_path):
            dataset_dir = base_path / dataset

            if not dataset_dir.exists():
                raise FileNotFoundError(f"Missing sequence file for {dataset_dir}!")

            if dataset not in file2taxon:
                raise FileNotFoundError(f"Missing taxon information for {dataset_dir}!")

        virus_task = AutoEvalTask(framework_name=self.get_framework_name(),
                                  dataset_name="virus",
                                  input_files=list(virus_taxon_files),
                                  type="Protein")
        non_virus_task = AutoEvalTask(framework_name=self.get_framework_name(),
                                      dataset_name="nonvirus",
                                      input_files=list(non_virus_taxon_files),
                                      type="Protein")
        # Total task contains all datasets and will re-use cached results
        total_task = AutoEvalTask(framework_name=self.get_framework_name(),
                                  dataset_name="total",
                                  input_files=list(virus_taxon_files) + list(non_virus_taxon_files),
                                  type="Protein")
        return [virus_task, non_virus_task, total_task]
