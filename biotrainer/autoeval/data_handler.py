from __future__ import annotations

import os
import shutil
import requests

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from appdirs import user_cache_dir
from abc import ABC, abstractmethod
from typing import List, Optional, Union


class AutoEvalDataHandler(ABC):

    def download_data(self, data_dir: Path) -> None:
        """
        Download and extract a data archive from a list of URLs, using them as fallbacks
        Urls can be a single URL or list of URLs to download the data from (will be tried in order)

        Args:
            data_dir: Directory to extract the data to
        """
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
                return  # Success - exit the function

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

    @staticmethod
    @abstractmethod
    def get_framework_name():
        raise NotImplementedError

    def get_framework_base_path(self, custom_framework_storage_path: Optional[Union[str, Path]] = None) -> Path:
        if custom_framework_storage_path:
            return Path(custom_framework_storage_path)
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
    def is_download_necessary(base_path: Path) -> bool:
        if not base_path.is_dir():
            raise ValueError(f"Given path {base_path} is not a directory!")
        return len(os.listdir(base_path)) == 0  # Directory is empty => Download

    @abstractmethod
    def preprocess(self, base_path: Path, min_seq_length: int, max_seq_length: int):
        raise NotImplementedError

    @abstractmethod
    def get_tasks(self, base_path: Path, min_seq_length: int, max_seq_length: int) -> List[AutoEvalTask]:
        raise NotImplementedError


@dataclass
class AutoEvalTask:
    name: str
    input_file: Path
    type: str
