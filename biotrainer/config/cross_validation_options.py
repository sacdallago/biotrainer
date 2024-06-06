from abc import ABC, abstractmethod
from typing import List, Union, Any, Final

from .config_option import ConfigOption, classproperty


class CrossValidationOption(ConfigOption, ABC):

    @classproperty
    def category(self) -> str:
        return "cross_validation_option"

    @classproperty
    @abstractmethod
    def cv_methods(self) -> List[str]:
        return []

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False


class Method(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "method"

    @classproperty
    def cv_methods(self) -> List[str]:
        return []

    @classproperty
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "hold_out"

    @property
    def possible_values(self) -> List[Any]:
        return ["hold_out", "k_fold", "leave_p_out"]

    def required(self) -> bool:
        return True


class K(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "k"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["k_fold"]

    def required(self) -> bool:
        return False

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return value >= 2


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

    @property
    def possible_values(self) -> List[Any]:
        return [True, False]

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

    def required(self) -> bool:
        return False

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return value >= 1


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

    @property
    def possible_values(self) -> List[Any]:
        return [True, False]

    def required(self) -> bool:
        return False


class NestedK(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "nested_k"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["k_fold"]

    def required(self) -> bool:
        return False

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return value >= 2


class SearchMethod(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "search_method"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["k_fold"]

    @property
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

    def required(self) -> bool:
        return False

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return value >= 2


class ChooseBy(CrossValidationOption):

    @classproperty
    def name(self) -> str:
        return "choose_by"

    @classproperty
    def cv_methods(self) -> List[str]:
        return ["hold_out", "k_fold", "leave_p_out"]

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "loss"

    @property
    def possible_values(self) -> List[Any]:
        return ["loss", "accuracy", "precision", "recall"]  # Add possible metrics to protocol enum

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

    def required(self) -> bool:
        return False

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return value >= 1


CROSS_VALIDATION_CONFIG_KEY: Final[str] = "cross_validation_config"
cross_validation_options: List = [Method, K, Stratified, Repeat, Nested, NestedK,
                                  SearchMethod, NMaxEvaluationsRandom, ChooseBy, P]
