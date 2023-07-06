from abc import ABC, abstractmethod
from typing import List, Type, Union, Any

from .config_option import ConfigOption, classproperty


class CrossValidationOption(ConfigOption, ABC):

    @classproperty
    def category(self) -> str:
        return "cross_validation_option"

    @abstractmethod
    @classproperty
    def methods(self) -> List[str]:
        return []


class Method(CrossValidationOption):

    def name(self) -> str:
        return "method"

    def methods(self) -> List[str]:
        return []

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "hold_out"

    def possible_values(self) -> List[Any]:
        return ["hold_out", "k_fold", "leave_p_out"]

    def possible_types(self) -> List[Type]:
        return [str]

    def required(self) -> bool:
        return True


class K(CrossValidationOption):

    def name(self) -> str:
        return "k"

    def methods(self) -> List[str]:
        return ["k_fold"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 5

    def possible_types(self) -> List[Type]:
        return [int]

    def required(self) -> bool:
        return False

    def is_value_valid(self) -> bool:
        return self.value >= 2


class Stratified(CrossValidationOption):

    def name(self) -> str:
        return "stratified"

    def methods(self) -> List[str]:
        return ["k_fold"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return False

    def possible_values(self) -> List[Any]:
        return [True, False]

    def possible_types(self) -> List[Type]:
        return [bool]

    def required(self) -> bool:
        return False


class Repeat(CrossValidationOption):

    def name(self) -> str:
        return "repeat"

    def methods(self) -> List[str]:
        return ["k_fold"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 1  # No repeat

    def possible_types(self) -> List[Type]:
        return [int]

    def required(self) -> bool:
        return False

    def is_value_valid(self) -> bool:
        return self.value >= 1


class Nested(CrossValidationOption):

    def name(self) -> str:
        return "nested"

    def methods(self) -> List[str]:
        return ["k_fold"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return False

    def possible_values(self) -> List[Any]:
        return [True, False]

    def possible_types(self) -> List[Type]:
        return [bool]

    def required(self) -> bool:
        return False


class NestedK(CrossValidationOption):

    def name(self) -> str:
        return "nested_k"

    def methods(self) -> List[str]:
        return ["k_fold"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 3

    def possible_types(self) -> List[Type]:
        return [int]

    def required(self) -> bool:
        return False

    def is_value_valid(self) -> bool:
        return self.value >= 2


class SearchMethod(CrossValidationOption):

    def name(self) -> str:
        return "search_method"

    def methods(self) -> List[str]:
        return ["k_fold"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    def possible_types(self) -> List[Type]:
        return [str]

    def required(self) -> bool:
        return False


class NMaxEvaluationsRandom(CrossValidationOption):

    def name(self) -> str:
        return "n_max_evaluations_random"

    def methods(self) -> List[str]:
        return ["k_fold"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 3

    def possible_types(self) -> List[Type]:
        return [int]

    def required(self) -> bool:
        return False

    def is_value_valid(self) -> bool:
        return self.value >= 2


class ChooseBy(CrossValidationOption):

    def name(self) -> str:
        return "choose_by"

    def methods(self) -> List[str]:
        return ["k_fold", "leave_p_out"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "loss"

    def possible_values(self) -> List[Any]:
        return []  # Add possible metrics to protocol enum

    def possible_types(self) -> List[Type]:
        return [str]

    def required(self) -> bool:
        return False


class P(CrossValidationOption):

    def name(self) -> str:
        return "p"

    def methods(self) -> List[str]:
        return ["leave_p_out"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 5

    def possible_types(self) -> List[Type]:
        return [int]

    def required(self) -> bool:
        return False

    def is_value_valid(self) -> bool:
        return self.value >= 1
