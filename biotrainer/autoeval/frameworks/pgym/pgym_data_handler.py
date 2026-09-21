import os
import pandas as pd

from pathlib import Path
from typing import List, Optional
from biotrainer_core.data_classes.autoeval import AutoEvalTask, DEV_MODE_INDICATOR

from ...core import AutoEvalDataHandler


class PGYMDataHandler(AutoEvalDataHandler):
    """Handles PGYM dataset related operations"""

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

    def get_tasks(self, base_path: Path, min_seq_length: Optional[int], max_seq_length: Optional[int]) \
            -> List[AutoEvalTask]:
        """Build tasks for all PGYM datasets"""
        virus_identifier = "virus"
        substitutions_path = base_path / "DMS_ProteinGym_substitutions"
        reference_df = pd.read_csv(self.get_reference_file_path(base_path))

        file2taxon = {row["DMS_id"] + ".csv": row["taxon"] for _, row in reference_df.iterrows()}
        file2devmode = {row["DMS_id"] + ".csv": row["pbc_dev_mode"] for _, row in reference_df.iterrows()}

        # Full mode (evaluation)
        virus_taxon_files = [substitutions_path / file for file, taxon in file2taxon.items()
                             if taxon.lower() == virus_identifier]
        non_virus_taxon_files = [substitutions_path / file for file, taxon in file2taxon.items()
                                 if taxon.lower() != virus_identifier]
        assert len(virus_taxon_files) + len(non_virus_taxon_files) == len(file2taxon)

        # Dev mode
        total_samples_dev_mode = sum([1 for mode in file2devmode.values() if mode])
        virus_taxon_files_dev_mode = [substitutions_path / file for file, devmode in file2devmode.items()
                                      if file2taxon[file].lower() == virus_identifier and devmode]
        non_virus_taxon_files_dev_mode = [substitutions_path / file for file, devmode in file2devmode.items()
                                          if file2taxon[file].lower() != virus_identifier and devmode]
        assert len(virus_taxon_files_dev_mode) + len(non_virus_taxon_files_dev_mode) == total_samples_dev_mode

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
        virus_task_dev = AutoEvalTask(framework_name=self.get_framework_name(),
                                      dataset_name="virus" + DEV_MODE_INDICATOR,
                                      input_files=list(virus_taxon_files_dev_mode),
                                      type="Protein")
        non_virus_task = AutoEvalTask(framework_name=self.get_framework_name(),
                                      dataset_name="nonvirus",
                                      input_files=list(non_virus_taxon_files),
                                      type="Protein")
        non_virus_task_dev = AutoEvalTask(framework_name=self.get_framework_name(),
                                          dataset_name="nonvirus" + DEV_MODE_INDICATOR,
                                          input_files=list(non_virus_taxon_files_dev_mode),
                                          type="Protein")
        # Total tasks contain all datasets and will re-use cached results
        total_task = AutoEvalTask(framework_name=self.get_framework_name(),
                                  dataset_name="total",
                                  input_files=list(virus_taxon_files) + list(non_virus_taxon_files),
                                  type="Protein")
        total_task_dev = AutoEvalTask(framework_name=self.get_framework_name(),
                                      dataset_name="total" +  DEV_MODE_INDICATOR,
                                      input_files=list(virus_taxon_files_dev_mode) + list(non_virus_taxon_files_dev_mode),
                                      type="Protein")
        return [virus_task, non_virus_task, total_task, virus_task_dev, non_virus_task_dev, total_task_dev]
