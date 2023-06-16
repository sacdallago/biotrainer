from abc import ABC, abstractmethod
from typing import List, Type, Union, Tuple

from .config_option import ConfigOption
from ..protocols import Protocol


class ConfigRule(ABC):
    @abstractmethod
    def apply(self, protocol: Protocol, config: List) -> Tuple[bool, str]:
        """
        Applies the config rule to the given protocol and config
        :param protocol: Chosen protocol
        :param config: Provided config
        :return: Tuple: [0] - True/False indicating if the rule holds for the config
                        [1] - Reason if the rule does not apply for the given config
        """
        raise NotImplementedError


class MutualExclusive(ConfigRule):

    def __init__(self, exclusive: List[Type]):
        self.exclusive_options = [exclusive_option.__class__ for exclusive_option in exclusive]

    def apply(self, protocol: Protocol, config: List[ConfigOption]) -> Tuple[bool, str]:
        occurrences = 0
        for config_option in config:
            if config_option.__class__ in self.exclusive_options:
                occurrences += 1

            if occurrences > 1:
                return False, f"{config_option} is mutual exclusive with {self.exclusive_options}."

        return True, ""


class ProtocolRequires(ConfigRule):

    def __init__(self, protocol: Union[Protocol, List[Protocol]], requires: List[Type]):
        if type(protocol) == Protocol:
            self._protocols = [protocol]
        else:
            self._protocols = protocol
        self._required_options = requires

    def apply(self, protocol: Protocol, config: List) -> Tuple[bool, str]:
        if protocol not in self._protocols:
            return True, ""
        else:
            config_classes = [config_option.__class__ for config_option in config]
            for required_option in self._required_options:
                if required_option not in config_classes:
                    return False, f"{protocol} requires {required_option(protocol).name} to be set."

            return True, ""
