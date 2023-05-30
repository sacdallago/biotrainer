import os
import typing
from typing import List, Union, Any, Type
from abc import ABC, abstractmethod

from ..protocols import Protocols


class ConfigurationException(Exception):
    """
    Exception for invalid configurations
    """


class ConfigOption(ABC):

    _protocol: Protocols = None

    def __init__(self, protocol: Protocols):
        self._protocol = protocol

    @property
    @abstractmethod
    def name(self) -> str:
        return "config_option"

    @property
    @abstractmethod
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "config_option"

    @property
    @abstractmethod
    def possible_types(self) -> List[Type]:
        return [Any]

    @property
    def possible_values(self) -> List[Any]:
        # List is not empty if this config option is restricted to certain values
        return []

    def is_value_valid(self, value: Any) -> bool:
        return value in self.possible_values

    @property
    def allowed_protocols(self) -> List[Protocols]:
        return Protocols.all()

    def to_dict(self):
        return {"name": str(self.name),
                "default_value": str(self.default_value),
                "possible_values": list(map(str, self.possible_values)),
                }


class FileOption(ConfigOption, ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        return "file_option"

    @property
    @abstractmethod
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "file_option"

    @property
    @abstractmethod
    def allowed_formats(self) -> List[str]:
        pass

    @property
    def allow_download(self) -> bool:
        return False

    @staticmethod
    def _validate_file(file_path: str):
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
            raise ConfigurationException(f"The configuration file at '{file_path}' does not exist") from e

    def is_value_valid(self, value: Any) -> bool:
        return self._validate_file(value)
