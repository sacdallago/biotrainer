from __future__ import annotations

import os
import logging
import shutil

from pathlib import Path
from urllib import request
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from typing import List, Union, Any

from ..protocols import Protocol

logger = logging.getLogger(__name__)


class ConfigurationException(Exception):
    """
    Exception for invalid configurations
    """


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class ConfigOption(ABC):
    _protocol: Protocol = None

    def __init__(self, protocol: Protocol, value: Any = None):
        self._protocol = protocol
        if value is None:
            value = self.default_value
        self.value = value

    @classproperty
    @abstractmethod
    def name(self) -> str:
        return "config_option"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @classproperty
    @abstractmethod
    def allow_multiple_values(self) -> bool:
        return False

    @property
    def possible_values(self) -> List[Any]:
        # List is not empty if this config option is restricted to certain values
        return []

    @classproperty
    @abstractmethod
    def required(self) -> bool:
        return False

    def is_list_option(self) -> bool:
        return ("range" in str(self.value) or type(self.value) is list or
                (type(self.value) is str and "[" in self.value and "]" in self.value))

    def check_value(self) -> bool:
        # Check for valid list option (range, list, list comprehension)
        if self.is_list_option():
            if not self.allow_multiple_values:
                return False

            value_eval = eval(str(self.value))
            value_type = type(value_eval)
            if value_type in [list, range]:
                return all([self._is_value_valid(self, value) for value in value_eval])
            else:
                raise ConfigurationException(f"Failed to parse list options for option {self.name}: {self.value}")
        return self._is_value_valid(self, self.value)

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return value in config_option.possible_values

    @classproperty
    def allowed_protocols(self) -> List[Protocol]:
        return Protocol.all()

    @classproperty
    def category(self) -> str:
        return "config_option"

    def transform_value_if_necessary(self, config_file_path: Path = None):
        pass

    def to_dict(self):
        return {"name": str(self.name),
                "category": str(self.category),
                "required": str(self.required),
                "default_value": str(self.default_value),
                "possible_values": list(map(str, self.possible_values)),
                }


class FileOption(ConfigOption, ABC):

    @classproperty
    @abstractmethod
    def name(self) -> str:
        return "file_option"

    @classproperty
    @abstractmethod
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @classproperty
    @abstractmethod
    def allowed_formats(self) -> List[str]:
        pass

    @classproperty
    def allow_download(self) -> bool:
        return False

    @classproperty
    def category(self) -> str:
        return "file_option"

    def _validate_file(self, file_path: str):
        """
        Verify if a file exists and is not empty.
        Parameters
        ----------
        file_path : str
            Path to file to check
        Returns
        -------
        bool
            True if file exists and is non-zero size,
            False otherwise.
        """
        try:
            if os.stat(file_path).st_size == 0:
                raise ConfigurationException(f"The file at '{file_path}' is empty")
            return True
        except (OSError, TypeError) as e:
            raise ConfigurationException(f"The {self.name} at '{file_path}' does not exist") from e

    @staticmethod
    def _is_url(value: str):
        return urlparse(value).scheme in ["http", "https", "ftp"]

    def _download_file_if_necessary(self, config_file_path: Path):
        url = self.value
        if self._is_url(url) and self.allow_download:
            try:
                logger.info(f"Trying to download {self.name} from {url}")
                req = request.Request(url, headers={
                    'User-Agent': "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"
                })

                file_name = url.split("/")[-1]
                save_path = str(config_file_path) + "/downloaded_" + file_name
                with request.urlopen(req) as response, open(save_path, 'wb') as outfile:
                    if response.status == 200:
                        logger.info(f"OK - Downloading file {self.name} (size: {response.length / pow(2, 20)} MB)..")
                    shutil.copyfileobj(response, outfile)
                logger.info(f"{self.name} successfully downloaded and stored at {save_path}.")
                self.value = save_path
            except Exception as e:
                raise Exception(f"Could not download {self.name} from url {url}") from e

    def _make_path_absolute_if_necessary(self, config_file_path: Path):
        absolute_path = (config_file_path / self.value).absolute()
        if absolute_path.is_file() or absolute_path.is_dir():
            self.value = str(absolute_path)

    def transform_value_if_necessary(self, config_file_path: Path = None):
        self._download_file_if_necessary(config_file_path)
        self._make_path_absolute_if_necessary(config_file_path)

    @staticmethod
    def _is_value_valid(config_option: FileOption, value) -> bool:
        if config_option._is_url(str(value)):
            return config_option.allow_download
        return config_option._validate_file(value)
