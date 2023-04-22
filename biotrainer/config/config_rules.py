from abc import ABC, abstractmethod
from typing import List, Type, Union

from .config_option import ConfigOption
from ..protocols import Protocols


class ConfigRule(ABC):
    @abstractmethod
    def apply(self, protocol: Protocols, config: List) -> bool:
        raise NotImplementedError


class MutualExclusive(ConfigRule):

    def __init__(self, exclusive: List[Type]):
        self.exclusive_options = [exclusive_option.__class__ for exclusive_option in exclusive]

    def apply(self, protocol: Protocols, config: List[ConfigOption]) -> bool:
        occurrences = 0
        for config_option in config:
            if config_option.__class__ in self.exclusive_options:
                occurrences += 1

            if occurrences > 1:
                return False


class ProtocolRequires(ConfigRule):

    def __init__(self, protocol: Union[Protocols, List[Protocols]], requires: List[Type]):
        if type(protocol) == Protocols:
            self._protocols = [protocol]
        else:
            self._protocols = protocol
        self._config_options = [config_option.__class__ for config_option in requires]

    def apply(self, protocol: Protocols, config: List) -> bool:
        if protocol not in self._protocols:
            return True
        else:
            config_classes = [config_option.__class__ for config_option in config]
            for config_option in self._config_options:
                if config_option not in config_classes:
                    return False

            return True


class ProtocolProhibits(ConfigRule):

    def __init__(self, protocol: Union[Protocols, List[Protocols]], prohibits: List[Type]):
        if type(protocol) == Protocols:
            self._protocols = [protocol]
        else:
            self._protocols = protocol
        self._config_options = [config_option.__class__ for config_option in prohibits]

    def apply(self, protocol: Protocols, config: List) -> bool:
        if protocol not in self._protocols:
            return True
        else:
            config_classes = [config_option.__class__ for config_option in config]
            for config_option in self._config_options:
                if config_option in config_classes:
                    return False

            return True
