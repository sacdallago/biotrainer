from typing import List, Type, Any, Union

from .config_option import ConfigOption
from ..protocols import Protocols

from ..models import get_available_models_dict
from ..losses import get_available_losses_dict
from ..optimizers import get_available_optimizers_dict


class ModelChoice(ConfigOption):

    @property
    def name(self) -> str:
        return "model_choice"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        if self._protocol == Protocols.residues_to_class:
            return "LightAttention"
        else:
            return "FNN"

    def is_value_valid(self, value) -> bool:
        return value in list(get_available_models_dict()[self._protocol].keys())

    @property
    def possible_types(self) -> List[Type]:
        return [str]


class OptimizerChoice(ConfigOption):

    @property
    def name(self) -> str:
        return "optimizer_choice"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "adam"

    def is_value_valid(self, value) -> bool:
        return value in list(get_available_optimizers_dict()[self._protocol].keys())

    @property
    def possible_types(self) -> List[Type]:
        return [str]


class LearningRate(ConfigOption):
    @property
    def name(self) -> str:
        return "learning_rate"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 1e-3

    def is_value_valid(self, value) -> bool:
        return 0.0 < value < 1.0

    @property
    def possible_types(self) -> List[Type]:
        return [float]


class Epsilon(ConfigOption):

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


class LossChoice(ConfigOption):
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

    def is_value_valid(self, value: Any) -> bool:
        return value in list(get_available_losses_dict()[self._protocol].keys())


model_options: List = [ModelChoice, OptimizerChoice, LearningRate, Epsilon, LossChoice]
