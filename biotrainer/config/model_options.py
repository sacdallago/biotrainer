from abc import ABC
from typing import List, Type, Any, Union

from .config_option import ConfigOption, classproperty
from ..protocols import Protocol

from ..models import get_available_models_dict
from ..losses import get_available_losses_dict
from ..optimizers import get_available_optimizers_dict


class ModelOption(ConfigOption, ABC):

    @classproperty
    def category(self) -> str:
        return "model_option"


class ModelChoice(ModelOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "model_choice"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        if self._protocol == Protocol.residues_to_class:
            return "LightAttention"
        else:
            return "FNN"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return True

    @property
    def possible_values(self) -> List[Any]:
        return list(get_available_models_dict()[self._protocol].keys())

    @classproperty
    def required(self) -> bool:
        return True


class OptimizerChoice(ModelOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "optimizer_choice"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "adam"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @property
    def possible_values(self) -> List[Any]:
        return list(get_available_optimizers_dict()[self._protocol].keys())

    @classproperty
    def required(self) -> bool:
        return False


class LearningRate(ModelOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "learning_rate"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 1e-3

    @classproperty
    def allow_multiple_values(self) -> bool:
        return True

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return 0.0 < value < 1.0

    @classproperty
    def required(self) -> bool:
        return False


class DropoutRate(ModelOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "dropout_rate"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 0.25

    @classproperty
    def allow_multiple_values(self) -> bool:
        return True

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return 0.0 < value < 1.0

    @classproperty
    def required(self) -> bool:
        return False


class Epsilon(ModelOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "epsilon"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 1e-3

    @classproperty
    def allow_multiple_values(self) -> bool:
        return True

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return 0.0 < value < 1.0

    @classproperty
    def required(self) -> bool:
        return False


class LossChoice(ModelOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "loss_choice"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        if self._protocol in Protocol.regression_protocols():
            return "mean_squared_error"
        else:
            return "cross_entropy_loss"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @property
    def possible_values(self) -> List[Any]:
        return list(get_available_losses_dict()[self._protocol].keys())

    @classproperty
    def required(self) -> bool:
        return False


model_options: List = [ModelChoice, OptimizerChoice, LearningRate, Epsilon, LossChoice]
