from abc import ABC, abstractmethod
from typing import List, Type, Union, Tuple, Any

from .config_option import ConfigOption
from .cross_validation_options import Method
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

    def __init__(self, exclusive: List, allowed_values: List[str] = None, error_message: str = ""):
        if allowed_values is None:
            allowed_values = []
        self.exclusive_options = exclusive
        self.allowed_values = allowed_values
        self.error_message = "\n" + error_message if error_message != "" else ""

    def apply(self, protocol: Protocol, config: List[ConfigOption]) -> Tuple[bool, str]:
        occurrences = 0
        for config_option in config:
            if config_option.__class__ in self.exclusive_options and config_option.value not in self.allowed_values:
                occurrences += 1

            if occurrences > 1:
                other_exclusive_options = [exclusive_option.name for exclusive_option in self.exclusive_options
                                           if exclusive_option != config_option.__class__]
                return False, f"{config_option.name} is mutual exclusive with {other_exclusive_options}." \
                              f"{self.error_message}"

        return True, ""


class ProtocolRequires(ConfigRule):

    def __init__(self, protocol: Union[Protocol, List[Protocol]], requires: List):
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
                    return False, f"{protocol} requires {required_option.name} to be set."

            return True, ""


class OptionValueRequires(ConfigRule):

    def __init__(self, option: Any, value: Any, requires: List):
        self._option = option
        self._value = value
        self._required_options = requires

    def apply(self, protocol: Protocol, config: List) -> Tuple[bool, str]:
        config_class_dict = {config_option.__class__: config_option.value for config_option in config}

        if self._option.__class__ in config_class_dict.keys() and \
                config_class_dict[self._option.__class__] == self._value:
            for required_option in self._required_options:
                if required_option not in config_class_dict.keys():
                    return False, f"{self._option.name} with value {self._value} requires {required_option.name} " \
                                  f"to be set."

        return True, ""
