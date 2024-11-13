from abc import ABC, abstractmethod
from typing import List, Union, Any, Final, Type

from .config_option import ConfigOption, classproperty


class CrossValidationOption(ConfigOption, ABC):
    """
    Abstract base class for cross-validation configuration options.

    Extends `ConfigOption` to provide a framework for defining cross-validation
    specific options, including supported methods and whether multiple values
    are allowed.
    """

    @classproperty
    def category(self) -> str:
        return "cross_validation_option"

    @classproperty
    @abstractmethod
    def cv_methods(self) -> List[str]:
        """
        List of cross-validation methods that the option is applicable to.

        Returns:
            List[str]: A list of supported cross-validation method names.
        """
        return []

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False


class Method(CrossValidationOption):
    """
    Configuration option for selecting the cross-validation method.

    This option allows users to specify the method of cross-validation to be used.
    It is a required option and supports predefined methods.
    """

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
    """
    Configuration option for specifying the number of folds in k-fold cross-validation.

    This option is applicable only to the "k_fold" cross-validation method and allows
    users to define the number of folds. It ensures that the value provided is valid.
    """

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
    """
    Configuration option to enable stratified k-fold cross-validation.

    When enabled, this option ensures that each fold maintains the percentage of samples for each class.
    """

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
    """
    Configuration option for specifying the number of repetitions in cross-validation.

    This option allows users to repeat the cross-validation process multiple times to obtain more robust estimates.
    """

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
    """
    Configuration option to enable nested cross-validation.

    Nested cross-validation is used to provide an unbiased evaluation of a model's performance.
    """

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
    """
    Configuration option for specifying the number of folds in nested k-fold cross-validation.

    This option is applicable only when nested cross-validation is enabled and the method is "k_fold".
    """

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
    """
    Configuration option for selecting the hyperparameter search method.

    This option allows users to choose between different hyperparameter optimization techniques.
    """

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
    """
    Configuration option for specifying the maximum number of evaluations in random search.

    This option limits the number of hyperparameter combinations to evaluate during random search.
    """

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
    """
    Configuration option for selecting the metric to choose the best model.

    This option allows users to specify which evaluation metric to use when selecting the best model.
    """

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
    """
    Configuration option for specifying the 'p' parameter in leave-p-out cross-validation.

    This option defines the number of samples to leave out in each iteration of leave-p-out cross-validation.
    """

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


# Constant key used to reference the cross-validation configuration.
CROSS_VALIDATION_CONFIG_KEY: Final[str] = "cross_validation_config"

# List of all cross-validation configuration option classes.
cross_validation_options: List[Type[CrossValidationOption]] = [
    Method,
    K,
    Stratified,
    Repeat,
    Nested,
    NestedK,
    SearchMethod,
    NMaxEvaluationsRandom,
    ChooseBy,
    P
]
