from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union, Any, Dict
from urllib import request
from urllib.parse import urlparse

from ..protocols import Protocol

logger = logging.getLogger(__name__)


class ConfigurationException(Exception):
    """
    Exception raised for invalid configurations.

    This exception is used to indicate errors related to configuration validation and processing.
    """


class classproperty(property):
    """
    A decorator to create class-level properties.

    Enables a method to be accessed as a property directly on the class without instantiation.
    """
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class ConfigOption(ABC):
    """
    Abstract base class for configuration options.

    Provides a framework for defining and validating configuration options
    within a protocol, including default values, possible values, and serialization.

    Attributes:
        _protocol (Protocol): The associated protocol for the configuration option.
        value (Any): The current value of the configuration option.
    """

    _protocol: Protocol = None

    def __init__(self, protocol: Protocol, value: Any = None):
        """
        Initialize a ConfigOption instance.

        Args:
            protocol (Protocol): The protocol associated with this configuration option.
            value (Any, optional): The initial value for the option. Defaults to the default value.
        """
        self._protocol = protocol
        if value is None:
            value = self.default_value
        self.value = value

    @classproperty
    @abstractmethod
    def name(self) -> str:
        """
        The name identifier for the configuration option.

        Must be overridden by subclasses to provide a unique name.

        Returns:
            str: The name of the configuration option.
        """
        return "config_option"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        """
        The default value for the configuration option.

        Can be overridden by subclasses to provide specific default values.

        Returns:
            Union[str, int, float, bool, Any]: The default value.
        """
        return ""

    @classproperty
    @abstractmethod
    def allow_multiple_values(self) -> bool:
        """
        Indicates if multiple values are allowed for this option.

        Must be overridden by subclasses to specify if the option can accept multiple values.

        Returns:
            bool: True if multiple values are allowed, False otherwise.
        """
        return False

    @property
    def possible_values(self) -> List[Any]:
        """
        A list of permissible values for the configuration option.

        Can be overridden by subclasses to restrict the allowed values.

        Returns:
            List[Any]: The possible values. Empty if unrestricted.
        """
        return []

    @classproperty
    @abstractmethod
    def required(self) -> bool:
        """
        Specifies whether the option is mandatory.

        Must be overridden by subclasses to indicate if the option is required.

        Returns:
            bool: True if the option is required, False otherwise.
        """
        return False

    def is_list_option(self) -> bool:
        """
        Determine if the option value is a list, range, or a valid range string.

        Useful for hyperparameter optimization.

        Returns:
            bool: True if `self.value` is a list, range, or a valid range string, else False.
        """
        return ("range" in str(self.value) or isinstance(self.value, list) or
                (isinstance(self.value, str) and "[" in self.value and "]" in self.value))

    @classproperty
    def is_file_option(self) -> bool:
        """
        Determine if the option pertains to file inputs or outputs.

        Returns:
            bool: True if category is 'file_option' or 'input_option', else False.
        """
        return self.category in {"file_option", "input_option"}

    def check_value(self) -> bool:
        """
        Validate the current value against possible constraints.

        Checks if the value is valid based on whether it's a list option and its possible values.

        Returns:
            bool: True if the value is valid, otherwise False.

        Raises:
            ConfigurationException: If list parsing fails for list options.
        """
        if self.is_list_option():
            if not self.allow_multiple_values:
                return False

            try:
                value_eval = eval(str(self.value))
            except Exception as e:
                raise ConfigurationException(f"Failed to evaluate the value for option {self.name}: {self.value}") from e

            value_type = type(value_eval)
            if value_type in [list, range]:
                return all([self._is_value_valid(self, value) for value in value_eval])
            else:
                raise ConfigurationException(f"Failed to parse list options for option {self.name}: {self.value}")
        return self._is_value_valid(self, self.value)

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        """
        Check if a value is within the possible values.

        Args:
            config_option (ConfigOption): The configuration option instance.
            value (Any): The value to validate.

        Returns:
            bool: True if valid, else False.
        """
        return value in config_option.possible_values

    @classproperty
    def allowed_protocols(self) -> List[Protocol]:
        """
        List the protocols that permit this configuration option.

        Returns:
            List[Protocol]: Allowed protocols.
        """
        return Protocol.all()

    @classproperty
    def category(self) -> str:
        """
        The category classification of the configuration option.

        Returns:
            str: The category name.
        """
        return "config_option"

    def transform_value_if_necessary(self, config_file_path: Path = None):
        """
        Transform the value if required. To be implemented by subclasses.

        Args:
            config_file_path (Path, optional): Path to the configuration file.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the configuration option to a dictionary.

        Returns:
            Dict[str, Any]: The serialized configuration option.
        """
        return {
            "name": str(self.name),
            "category": str(self.category),
            "required": str(self.required),
            "default_value": str(self.default_value),
            "possible_values": list(map(str, self.possible_values)),
        }


class FileOption(ConfigOption, ABC):
    """
    Abstract base class for file-based configuration options.

    Extends `ConfigOption` to handle file-specific operations such as validation,
    downloading from URLs, and path management.

    Subclasses must implement the following abstract properties:
        - `name`: The identifier for the configuration option.
        - `default_value`: The default value for the option.
        - `allowed_formats`: A list of permissible file formats.

    Attributes:
        None
    """

    @classproperty
    @abstractmethod
    def name(self) -> str:
        """
        The name identifier for the configuration option.

        Must be overridden by subclasses to provide a unique name.

        Returns:
            str: The name of the configuration option.
        """
        return "file_option"

    @classproperty
    @abstractmethod
    def default_value(self) -> Union[str, int, float, bool, Any]:
        """
        The default value for the configuration option.

        Must be overridden by subclasses to provide a specific default value.

        Returns:
            Union[str, int, float, bool, Any]: The default value.
        """
        return ""

    @classproperty
    @abstractmethod
    def allowed_formats(self) -> List[str]:
        """
        List of allowed file formats for the configuration option.

        Must be overridden by subclasses to specify permissible file extensions.

        Returns:
            List[str]: A list of allowed file format extensions.
        """
        pass

    @classproperty
    def allow_download(self) -> bool:
        """
        Indicates if downloading from URLs is permitted.

        Can be overridden by subclasses to allow or disallow downloads.

        Returns:
            bool: True if downloading is allowed, False otherwise.
        """
        return False

    @classproperty
    def category(self) -> str:
        """
        The category classification of the configuration option.

        Overrides the base class to specify 'file_option'.

        Returns:
            str: The category name, set to "file_option".
        """
        return "file_option"

    def _validate_file(self, file_path: str) -> bool:
        """
        Verify if a file exists and is not empty.

        Args:
            file_path (str): Path to the file to check.

        Returns:
            bool: True if the file exists and is non-empty.

        Raises:
            ConfigurationException: If the file does not exist or is empty.
        """
        try:
            if os.stat(file_path).st_size == 0:
                raise ConfigurationException(f"The file at '{file_path}' is empty")
            return True
        except (OSError, TypeError) as e:
            raise ConfigurationException(f"The {self.name} at '{file_path}' does not exist") from e

    @staticmethod
    def _is_url(value: str) -> bool:
        """
        Check if the given value is a valid URL.

        Args:
            value (str): The value to check.

        Returns:
            bool: True if `value` is a URL with scheme http, https, or ftp, else False.
        """
        return urlparse(value).scheme in ["http", "https", "ftp"]

    def _download_file_if_necessary(self, config_file_path: Path):
        """
        Download the file from a URL if downloading is allowed.

        Args:
            config_file_path (Path): Path to the configuration file directory.

        Raises:
            Exception: If the file cannot be downloaded.
        """
        url = self.value
        if self._is_url(url) and self.allow_download:
            try:
                logger.info(f"Trying to download {self.name} from {url}")
                req = request.Request(url, headers={
                    'User-Agent': "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"
                })

                file_name = url.split("/")[-1]
                save_path = str(config_file_path / f"downloaded_{file_name}")
                with request.urlopen(req) as response, open(save_path, 'wb') as outfile:
                    if response.status == 200:
                        logger.info(f"OK - Downloading file {self.name} (size: {response.length / (2**20):.2f} MB)..")
                    shutil.copyfileobj(response, outfile)
                logger.info(f"{self.name} successfully downloaded and stored at {save_path}.")
                self.value = save_path
            except Exception as e:
                raise Exception(f"Could not download {self.name} from url {url}") from e

    def _make_path_absolute_if_necessary(self, config_file_path: Path):
        """
        Convert the file path to an absolute path if it exists.

        Args:
            config_file_path (Path): Path to the configuration file directory.
        """
        absolute_path = (config_file_path / self.value).absolute()
        if absolute_path.is_file() or absolute_path.is_dir():
            self.value = str(absolute_path)

    def transform_value_if_necessary(self, config_file_path: Path = None):
        """
        Perform necessary transformations on the file path, such as downloading and making it absolute.

        Args:
            config_file_path (Path, optional): Path to the configuration file directory.
        """
        self._download_file_if_necessary(config_file_path)
        self._make_path_absolute_if_necessary(config_file_path)

    @staticmethod
    def _is_value_valid(config_option: FileOption, value: str) -> bool:
        """
        Validate the file value.

        Args:
            config_option (FileOption): The configuration option instance.
            value (str): The file path or URL to validate.

        Returns:
            bool: True if valid, else False.
        """
        if config_option._is_url(str(value)):
            return config_option.allow_download
        return config_option._validate_file(value)
