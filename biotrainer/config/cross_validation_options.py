from abc import ABC, abstractmethod
from typing import List, Type, Union, Any, Final

from .config_option import ConfigOption, classproperty


class CrossValidationOption(ConfigOption, ABC):

    @classproperty
    def category(self) -> str:
        return "cross_validation_option"

    @classproperty
    @abstractmethod
    def cv_methods(self) -> List[str]:
        return []


class Method(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "method"

    @classproperty
    def cv_methods(self) -> List[str]:
        return []

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "hold_out"

    @classproperty
    def possible_values(self) -> List[Any]:
        return ["hold_out", "k_fold", "leave_p_out"]

    @classproperty
    def possible_types(self) -> List[Type]:
        return [str]

    def required(self) -> bool:
        return True


class K(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "k"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["k_fold"]

    @classproperty
    def possible_types(self) -> List[Type]:
        return [int]

    def required(self) -> bool:
        return False

    def is_value_valid(self) -> bool:
        return self.value >= 2


class Stratified(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "stratified"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["k_fold"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return False

    @classproperty
    def possible_values(self) -> List[Any]:
        return [True, False]

    @classproperty
    def possible_types(self) -> List[Type]:
        return [bool]

    def required(self) -> bool:
        return False


class Repeat(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "repeat"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["k_fold"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 1  # No repeat

    @classproperty
    def possible_types(self) -> List[Type]:
        return [int]

    def required(self) -> bool:
        return False

    def is_value_valid(self) -> bool:
        return self.value >= 1


class Nested(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "nested"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["k_fold"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return False

    @classproperty
    def possible_values(self) -> List[Any]:
        return [True, False]

    @classproperty
    def possible_types(self) -> List[Type]:
        return [bool]

    def required(self) -> bool:
        return False


class NestedK(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "nested_k"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["k_fold"]

    @classproperty
    def possible_types(self) -> List[Type]:
        return [int]

    def required(self) -> bool:
        return False

    def is_value_valid(self) -> bool:
        return self.value >= 2


class SearchMethod(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "search_method"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["k_fold"]

    @classproperty
    def possible_types(self) -> List[Type]:
        return [str]

    @classproperty
    def possible_values(self) -> List[Any]:
        return ["random_search", "grid_search"]

    def required(self) -> bool:
        return False


class NMaxEvaluationsRandom(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "n_max_evaluations_random"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["k_fold"]

    @classproperty
    def possible_types(self) -> List[Type]:
        return [int]

    def required(self) -> bool:
        return False

    def is_value_valid(self) -> bool:
        return self.value >= 2


class ChooseBy(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "choose_by"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["k_fold", "leave_p_out"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "loss"

    @classproperty
    def possible_values(self) -> List[Any]:
        return []  # Add possible metrics to protocol enum

    @classproperty
    def possible_types(self) -> List[Type]:
        return [str]

    def required(self) -> bool:
        return False


class P(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "p"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["leave_p_out"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 5

    @classproperty
    def possible_types(self) -> List[Type]:
        return [int]

    def required(self) -> bool:
        return False

    def is_value_valid(self) -> bool:
        return self.value >= 1


CROSS_VALIDATION_CONFIG_KEY: Final[str] = "cross_validation_config"
cross_validation_options: List = [Method, K, Stratified, Repeat, Nested, NestedK,
                                  SearchMethod, NMaxEvaluationsRandom, ChooseBy, P]
