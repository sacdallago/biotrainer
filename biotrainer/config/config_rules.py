from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Any, Type, Dict

from .config_option import ConfigOption
from ..protocols import Protocol


class ConfigRule(ABC):
    @abstractmethod
    def apply(
            self,
            protocol: Protocol,
            config: List[ConfigOption],
            ignore_file_checks: bool
    ) -> Tuple[bool, str]:
        """
        Applies the configuration rule to the given protocol and configuration.

        Args:
            protocol (Protocol): The chosen protocol to which the rule is applied.
            config (List[ConfigOption]): A list of configuration options to validate against the rule.
            ignore_file_checks (bool): Flag indicating whether to ignore file-related checks.

        Returns:
            Tuple[bool, str]: A tuple where the first element is a boolean indicating if the rule holds,
                              and the second element is a string providing the reason if the rule does not hold.
        """
        raise NotImplementedError


class MutualExclusive(ConfigRule):
    """
    Enforces mutual exclusivity among specified configuration options.

    This rule ensures that no more than one of the specified exclusive options is set
    in the configuration. Optionally, it allows certain values to coexist even if they
    are among the exclusive options.

    Attributes:
        exclusive_options (List[Type[ConfigOption]]): 
            A list of `ConfigOption` classes that are mutually exclusive.
        allowed_values (List[str]): 
            A list of values that are permitted to coexist even if their corresponding options are exclusive.
        error_message (str): 
            An optional custom error message to include when the rule is violated.
    """

    def __init__(
            self,
            exclusive: List[Type[ConfigOption]], 
            allowed_values: List[str] = None, 
            error_message: str = ""
    ):
        """
        Initializes the MutualExclusive rule.

        Args:
            exclusive (List[Type[ConfigOption]]): 
                A list of `ConfigOption` classes that should be mutually exclusive.
            allowed_values (List[str], optional): 
                Optional list of string values that are allowed even if their corresponding options are exclusive.
                Defaults to an empty list if not provided.
            error_message (str, optional): 
                Optional custom error message to display when the rule is violated.
        """
        self.exclusive_options = exclusive
        self.allowed_values = allowed_values if allowed_values is not None else []
        self.error_message = f"\n{error_message}" if error_message else ""

    def apply(
            self,
            protocol: Protocol,
            config: List[ConfigOption],
            ignore_file_checks: bool
    ) -> Tuple[bool, str]:
        """
        Applies the mutual exclusivity rule to the given configuration.

        This method checks the provided configuration to ensure that no more than one of the
        mutually exclusive options is set, considering any allowed values. If `ignore_file_checks`
        is True and any of the exclusive options are file-related, the rule is automatically satisfied.

        Args:
            protocol (Protocol): The protocol to which the configuration is applied.
            config (List[ConfigOption]): A list of `ConfigOption` instances representing the current configuration.
            ignore_file_checks (bool): Flag indicating whether to ignore file-related checks in the rule.

        Returns:
            Tuple[bool, str]: A tuple where the first element is a boolean indicating if the rule holds,
                              and the second element is a string providing the reason if the rule does not hold.
        """
        if ignore_file_checks and any(option.is_file_option for option in self.exclusive_options):
            return True, ""

        occurrences = 0
        for config_option in config:
            if (config_option.__class__ in self.exclusive_options and
                config_option.value not in self.allowed_values
            ):
                occurrences += 1

            if occurrences > 1:
                other_exclusive_options = [
                    exclusive_option.name
                    for exclusive_option in self.exclusive_options
                    if exclusive_option != config_option.__class__
                ]
                return (
                    False,
                    f"{config_option.name} is mutual exclusive with {other_exclusive_options}. "
                    f"{self.error_message}"
                )

        return True, ""


class MutualExclusiveValues(ConfigRule):
    """
    Enforces mutual exclusivity based on specific values of configuration options.

    This rule ensures that certain values of specified configuration options cannot coexist
    within the configuration. It allows for fine-grained control over option value combinations.

    Attributes:
        exclusive_options (Dict[Type[ConfigOption], Any]): 
            A dictionary mapping `ConfigOption` classes to their exclusive values.
        error_message (str): 
            An optional custom error message to include when the rule is violated.
    """

    def __init__(
            self, 
            exclusive: Dict[Type[ConfigOption], Any], 
            error_message: str = ""
    ):
        """
        Initializes the MutualExclusiveValues rule.

        Args:
            exclusive (Dict[Type[ConfigOption], Any]): 
                A dictionary where keys are `ConfigOption` classes and values are the specific values
                that are mutually exclusive.
            error_message (str, optional): 
                Optional custom error message to display when the rule is violated.
        """
        self.exclusive_options = exclusive
        self.error_message = f"\n{error_message}" if error_message else ""

    def apply(
            self,
            protocol: Protocol,
            config: List[ConfigOption], 
            ignore_file_checks: bool
    ) -> Tuple[bool, str]:
        """
        Applies the mutual exclusivity rule based on specific option values to the given configuration.

        This method checks the provided configuration to ensure that no more than one of the
        specified option-value pairs is set. If `ignore_file_checks` is True and any of the exclusive
        options are file-related, the rule is automatically satisfied.

        Args:
            protocol (Protocol): The protocol to which the configuration is applied.
            config (List[ConfigOption]): A list of `ConfigOption` instances representing the current configuration.
            ignore_file_checks (bool): Flag indicating whether to ignore file-related checks in the rule.

        Returns:
            Tuple[bool, str]: A tuple where the first element is a boolean indicating if the rule holds,
                              and the second element is a string providing the reason if the rule does not hold.
        """
        if ignore_file_checks and any(option.is_file_option for option in self.exclusive_options):
            return True, ""

        occurrences = 0
        for config_option in config:
            if (
                config_option.__class__ in self.exclusive_options.keys() and 
                config_option.value == self.exclusive_options[config_option.__class__]
            ):
                occurrences += 1

            if occurrences > 1:
                other_exclusive_options = {
                    option_class.name: option_value 
                    for option_class, option_value in self.exclusive_options.items()
                    if option_class != config_option.__class__
                }
                return (
                    False, 
                    f"{config_option.name}: {config_option.value} is mutual exclusive "
                    f"with {other_exclusive_options}. {self.error_message}"
                )

        return True, ""


class ProtocolRequires(ConfigRule):
    """
    Enforces that specific protocols require certain configuration options.

    This rule ensures that when a particular protocol is selected, certain configuration
    options must be present in the configuration. It is useful for enforcing protocol-specific
    dependencies or prerequisites.

    Attributes:
        _protocols (List[Protocol]): 
            A list of protocols that require the specified configuration options.
        _required_options (List[Type[ConfigOption]]): 
            A list of `ConfigOption` classes that are required when the associated protocols are used.
    """

    def __init__(
            self,
            protocol: Union[Protocol,List[Protocol]],
            requires: List[Type[ConfigOption]]
    ):
        """
        Initializes the ProtocolRequires rule.

        Args:
            protocol (Union[Protocol, List[Protocol]]): 
                A single `Protocol` instance or a list of `Protocol` instances that trigger the requirement.
            requires (List[Type[ConfigOption]]): 
                A list of `ConfigOption` classes that are required when the specified protocol(s) are used.
        """
        if isinstance(protocol, Protocol):
            self._protocols = [protocol]
        else:
            self._protocols = protocol
        self._required_options = requires

    def apply(
            self,
            protocol: Protocol,
            config: List[ConfigOption],
            ignore_file_checks: bool
    ) -> Tuple[bool, str]:
        """
        Applies the protocol requirement rule to the given configuration.

        This method checks if the selected protocol is among the protocols that require certain
        configuration options. If so, it verifies that all required options are present in the
        configuration. If `ignore_file_checks` is True and any of the required options are file-related,
        the rule is automatically satisfied.

        Args:
            protocol (Protocol): The protocol to which the configuration is applied.
            config (List[ConfigOption]): A list of `ConfigOption` instances representing the current configuration.
            ignore_file_checks (bool): Flag indicating whether to ignore file-related checks in the rule.

        Returns:
            Tuple[bool, str]: A tuple where the first element is a boolean indicating if the rule holds,
                              and the second element is a string providing the reason if the rule does not hold.
        """        
        if ignore_file_checks and any(option.is_file_option for option in self._required_options):
            return True, ""

        if protocol not in self._protocols:
            return True, ""
        else:
            config_classes = [config_option.__class__ for config_option in config]
            for required_option in self._required_options:
                if required_option not in config_classes:
                    return False, f"{protocol} requires {required_option.name} to be set."

            return True, ""


class OptionValueRequires(ConfigRule):
    """
    Enforces that a specific value of a configuration option requires other options to be set.

    This rule ensures that when a configuration option is set to a particular value, certain
    other configuration options must also be present. It is useful for establishing dependencies
    between configuration options based on their values.

    Attributes:
        _option (Type[ConfigOption]): 
            The `ConfigOption` class that triggers the requirement when set to a specific value.
        _value (Any): 
            The specific value of `_option` that activates the requirement.
        _required_options (List[Type[ConfigOption]]): 
            A list of `ConfigOption` classes that are required when `_option` is set to `_value`.
    """

    def __init__(
            self,
            option: Any,
            value: Any,
            requires: List
    ):
        """
        Initializes the OptionValueRequires rule.

        Args:
            option (Type[ConfigOption]): 
                The `ConfigOption` class that, when set to a specific value, requires other options.
            value (Any): 
                The specific value of `option` that activates the requirement.
            requires (List[Type[ConfigOption]]): 
                A list of `ConfigOption` classes that must be set when `option` is set to `value`.
        """
        self._option = option
        self._value = value
        self._required_options = requires

    def apply(
            self,
            protocol: Protocol,
            config: List[ConfigOption],
            ignore_file_checks: bool
    ) -> Tuple[bool, str]:
        """
        Applies the option value requirement rule to the given configuration.

        This method checks if the specified configuration option is set to the required value.
        If so, it verifies that all dependent configuration options are present in the configuration.
        If `ignore_file_checks` is True and any of the required options are file-related,
        the rule is automatically satisfied.

        Args:
            protocol (Protocol): The protocol to which the configuration is applied.
            config (List[ConfigOption]): A list of `ConfigOption` instances representing the current configuration.
            ignore_file_checks (bool): Flag indicating whether to ignore file-related checks in the rule.

        Returns:
            Tuple[bool, str]: A tuple where the first element is a boolean indicating if the rule holds,
                              and the second element is a string providing the reason if the rule does not hold.
        """
        if ignore_file_checks and any(option.is_file_option for option in self._required_options):
            return True, ""

        config_class_dict = {config_option.name: config_option.value for config_option in config}

        if (
            self._option.name in config_class_dict and 
            config_class_dict[self._option.name] == self._value
        ):
            for required_option in self._required_options:
                if required_option.name not in config_class_dict:
                    return (
                        False,
                        f"{self._option.name} with value {self._value} requires {required_option.name} to be set."
                    )

        return True, ""


class AllowHyperparameterOptimization(ConfigRule):
    """
    Controls the allowance of hyperparameter optimization based on specific configuration options.

    This rule ensures that hyperparameter optimization is only enabled when certain configuration
    options are set to values that permit it. It also prevents hyperparameter optimization when
    specific options are set to values that disallow it.

    Attributes:
        _option (Type[ConfigOption]): 
            The `ConfigOption` class that controls the allowance of hyperparameter optimization.
        _value (Any): 
            The specific value of `_option` that permits or disallows hyperparameter optimization.
    """

    def __init__(
            self,
            option: Type[ConfigOption],
            value: Any
    ):
        """
        Initializes the AllowHyperparameterOptimization rule.

        Args:
            option (Type[ConfigOption]): 
                The `ConfigOption` class that determines the allowance of hyperparameter optimization.
            value (Any): 
                The specific value of `option` that permits or disallows hyperparameter optimization.
        """
        self._option = option
        self._value = value

    def apply(
            self,
            protocol: Protocol,
            config: List,
            ignore_file_checks: bool
    ) -> Tuple[bool, str]:
        """
        Applies the hyperparameter optimization allowance rule to the given configuration.

        This method checks whether hyperparameter optimization is allowed based on the value
        of a specific configuration option. It ensures that if the option is set to a value
        that permits optimization, at least one hyperparameter is being optimized. Conversely,
        if the option's value disallows optimization, it ensures that no hyperparameters are being optimized.

        Args:
            protocol (Protocol): The protocol to which the configuration is applied.
            config (List[ConfigOption]): A list of `ConfigOption` instances representing the current configuration.
            ignore_file_checks (bool): Flag indicating whether to ignore some checks in the rule.

        Returns:
            Tuple[bool, str]: A tuple where the first element is a boolean indicating if the rule holds,
                              and the second element is a string providing the reason if the rule does not hold.
        """
        config_class_dict = {config_option.name: config_option for config_option in config}
        all_list_options = [config_option.is_list_option() for config_option in config_class_dict.values()]

        if self._option.name in config_class_dict:
            if config_class_dict[self._option.name].value == self._value:
                if not any(all_list_options):
                    return (
                        False,
                        f"{self._option.name}={self._option.value} requires at least "
                        f"one hyperparameter to be optimized. Provide "
                        f"such a parameter via a list, list comprehension or range expression."
                    )
            else:
                if any(all_list_options):
                    return (
                        False, 
                        f"{self._option.name}={self._option.value} does not allow for hyperparameter optimization."
                    )
        else:
            if any(all_list_options):
                return (
                    False,
                    f"Hyperparameter optimization not allowed if {self._option} is missing!"
                )
            
        return True, ""
