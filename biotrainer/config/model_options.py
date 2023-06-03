from abc import ABC
from typing import List, Type, Any, Union

from .config_option import ConfigOption
from ..protocols import Protocols

from ..models import get_available_models_dict
from ..losses import get_available_losses_dict
from ..optimizers import get_available_optimizers_dict


class ModelOption(ConfigOption, ABC):

    @property
    def category(self) -> str:
        return "model_option"


class ModelChoice(ModelOption, ConfigOption):

    @property
    def name(self) -> str:
        return "model_choice"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        if self._protocol == Protocols.residues_to_class:
            return "LightAttention"
        else:
            return "FNN"

    @property
    def possible_types(self) -> List[Type]:
        return [str]

    @property
    def possible_values(self) -> List[Any]:
        return list(get_available_models_dict()[self._protocol].keys())

    @property
    def required(self) -> bool:
        return True


class OptimizerChoice(ModelOption, ConfigOption):

    @property
    def name(self) -> str:
        return "optimizer_choice"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "adam"

    @property
    def possible_types(self) -> List[Type]:
        return [str]

    @property
    def possible_values(self) -> List[Any]:
        return list(get_available_optimizers_dict()[self._protocol].keys())

    @property
    def required(self) -> bool:
        return False


class LearningRate(ModelOption, ConfigOption):
    @property
    def name(self) -> str:
        return "learning_rate"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 1e-3

    @property
    def possible_types(self) -> List[Type]:
        return [float]

    def is_value_valid(self, value) -> bool:
        return 0.0 < value < 1.0

    @property
    def required(self) -> bool:
        return False


class Epsilon(ModelOption, ConfigOption):

    @property
    def name(self) -> str:
        return "epsilon"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 1e-3

    @property
    def possible_types(self) -> List[Type]:
        return [float]

    def is_value_valid(self, value: Any) -> bool:
        return 0.0 < value < 1.0

    @property
    def required(self) -> bool:
        return False


class LossChoice(ModelOption, ConfigOption):
    @property
    def name(self) -> str:
        return "loss_choice"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        if self._protocol in Protocols.regression_protocols():
            return "mean_squared_error"
        else:
            return "cross_entropy_loss"

    @property
    def possible_types(self) -> List[Type]:
        return [str]

    @property
    def possible_values(self) -> List[Any]:
        return list(get_available_losses_dict()[self._protocol].keys())

    @property
    def required(self) -> bool:
        return False


model_options: List = [ModelChoice, OptimizerChoice, LearningRate, Epsilon, LossChoice]
