from __future__ import annotations

import os
import shutil
import requests

from tqdm import tqdm
from pathlib import Path
from appdirs import user_cache_dir
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from .autoeval_task import AutoEvalTask

from ...input_files import filter_FASTA


class AutoEvalDataHandler(ABC):

    def download_data(self, data_dir: Path) -> None:
        """
        Download and extract a data archive from a list of URLs, using them as fallbacks
        Urls can be a single URL or list of URLs to download the data from (will be tried in order)

        Args:
            data_dir: Directory to extract the data to
        """
        # Download data archive
        urls = self.get_download_urls()

        zip_file = data_dir.with_suffix('.zip')
        headers = {
            'Accept': 'application/zip, application/octet-stream',
            'User-Agent': 'biotrainer/autoeval'
        }

        for i, url in enumerate(urls, 1):
            try:
                print(f"Attempting download from {url} (attempt {i}/{len(urls)})..")
                response = requests.get(url, headers=headers, stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192  # 8 KB

                with open(zip_file, "wb") as f, tqdm(
                        desc="Downloading data",
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as progress_bar:
                    for data in response.iter_content(block_size):
                        size = f.write(data)
                        progress_bar.update(size)

                print("Unpacking data archive..")
                shutil.unpack_archive(zip_file, data_dir)

                # Remove the zip file after successful extraction
                zip_file.unlink()

                print("Data downloaded and unpacked successfully!")
                break  # Exit download loop
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
                if zip_file.exists():
                    zip_file.unlink()

                # If this was the last URL, raise the exception
                if i == len(urls):
                    print("All download attempts failed")
                    raise Exception("Failed to download data from all provided URLs") from e

                # Otherwise, continue to the next URL
                print("Trying next fallback URL...")
                continue

        # Download reference file if necessary
        reference_urls = self.get_reference_file_urls()
        if len(reference_urls) > 0:
            reference_file_path = self.get_reference_file_path(data_dir)
            if reference_file_path.exists():
                print(f"Reference file already exists at: {reference_file_path}")
                return
            if not reference_file_path.parent.exists():
                reference_file_path.parent.mkdir(parents=True)

            print("Downloading reference file..")
            for reference_url in reference_urls:
                try:
                    response = requests.get(reference_url, headers=headers, stream=False)
                    response.raise_for_status()

                    with open(reference_file_path, 'wb') as f:
                        f.write(response.content)

                    print(f"Reference file downloaded successfully: {reference_url}")
                    break
                except Exception as e:
                    print(f"Failed to download reference file from {reference_url}: {e}")

    @staticmethod
    @abstractmethod
    def get_framework_name():
        raise NotImplementedError

    @staticmethod
    def clear_autoeval_cache():
        shutil.rmtree(Path(user_cache_dir('biotrainer')) / "autoeval", ignore_errors=True)

    def get_framework_base_path(self, custom_storage_path: Optional[Union[str, Path]] = None) -> Path:
        if custom_storage_path:
            return Path(custom_storage_path) / self.get_framework_name()
        return Path(user_cache_dir('biotrainer')) / "autoeval" / self.get_framework_name()

    @staticmethod
    def get_preprocessed_file_dir_name(min_seq_length: int, max_seq_length: int):
        assert isinstance(min_seq_length, int) and isinstance(max_seq_length, int), \
            f"{min_seq_length} and {max_seq_length} must be integers"
        return f"preprocessed_{min_seq_length}_{max_seq_length}"

    @staticmethod
    @abstractmethod
    def get_download_urls():
        raise NotImplementedError

    @staticmethod
    def get_reference_file_urls() -> List[str]:
        return []

    @staticmethod
    def get_reference_file_name() -> str:
        return "reference.csv"

    def get_reference_file_path(self, base_path: Path) -> Path:
        return base_path / "reference" / self.get_reference_file_name()

    @staticmethod
    def is_download_necessary(base_path: Path) -> bool:
        if not base_path.is_dir():
            raise ValueError(f"Given path {base_path} is not a directory!")
        return len(os.listdir(base_path)) == 0  # Directory is empty => Download

    @abstractmethod
    def preprocess(self, base_path: Path, min_seq_length: Optional[int], max_seq_length: Optional[int]):
        raise NotImplementedError

    @abstractmethod
    def get_tasks(self, base_path: Path, min_seq_length: Optional[int], max_seq_length: Optional[int]) -> List[AutoEvalTask]:
        """
        Get tasks to execute in the autoeval pipeline via biotrainer.

        This is the core method of the framework data handler.
        """
        raise NotImplementedError

    @staticmethod
    def _get_input_file_path(dataset_dir: Path, name: str, min_seq_length: int, max_seq_length: int) -> Path:
        """Get the appropriate input file path (preprocessed if available)"""
        raw_path = dataset_dir / f"{name}.fasta"
        preprocessed_dir_name = AutoEvalDataHandler.get_preprocessed_file_dir_name(min_seq_length=min_seq_length,
                                                                                   max_seq_length=max_seq_length)
        preprocessed_path = dataset_dir / preprocessed_dir_name / f"{name}.fasta"

        if preprocessed_path.exists():
            return preprocessed_path
        return raw_path

    @staticmethod
    def _ensure_preprocessed_file(dataset_dir: Path, name: str, min_seq_length: int, max_seq_length: int) -> Path:
        """Ensure a preprocessed version of the file exists and return its path"""
        download_path = dataset_dir / f"{name}.fasta"
        preprocessed_dir_name = AutoEvalDataHandler.get_preprocessed_file_dir_name(min_seq_length=min_seq_length,
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
