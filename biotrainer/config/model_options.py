from abc import ABC
from typing import List, Any, Union, Type

from .config_option import ConfigOption, classproperty
from ..losses import get_available_losses_dict
from ..models import get_available_models_dict
from ..optimizers import get_available_optimizers_dict
from ..protocols import Protocol


class ModelOption(ConfigOption, ABC):
    """
    Abstract base class for model-related configuration options.

    Extends `ConfigOption` to provide a specialized framework for model options
    within a protocol. This class serves as a foundation for specific model options
    by setting common attributes and behaviors.
    """

    @classproperty
    def category(self) -> str:
        return "model_option"


class ModelChoice(ModelOption, ConfigOption):
    """
    Configuration option for selecting the model architecture.

    This option allows users to choose a specific model architecture from the available
    predefined models. It supports multiple selections if allowed and ensures that the
    selected models are valid for the chosen protocol.
    """

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
    """
    Configuration option for selecting the optimizer.

    This option allows users to choose an optimizer from the available predefined optimizers.
    It ensures that the selected optimizer is valid for the chosen protocol.
    """

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
    """
    Configuration option for setting the learning rate.

    This option allows users to define one or more learning rates for the optimizer.
    It ensures that the provided learning rates are within a valid range.
    """

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
    """
    Configuration option for setting the dropout rate.

    This option allows users to define one or more dropout rates to be applied in the model.
    It ensures that the provided dropout rates are within a valid range to prevent overfitting.
    """

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
    """
    Configuration option for setting the epsilon value.

    This option allows users to define one or more epsilon values used in optimization algorithms.
    It ensures that the provided epsilon values are within a valid range to maintain numerical stability.
    """

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
    """
    Configuration option for selecting the loss function.

    This option allows users to choose a specific loss function from the available
    predefined losses. The default loss function is determined based on whether the
    selected protocol is for regression or classification tasks.
    """

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


class DisablePytorchCompile(ModelOption, ConfigOption):
    """
    Configuration option for disabling PyTorch's compilation feature.

    This option allows users to disable the automatic compilation of PyTorch models,
    which can be useful for debugging or when encountering compatibility issues.
    """

    @classproperty
    def name(self) -> str:
        return "disable_pytorch_compile"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return True

    @property
    def possible_values(self) -> List[Any]:
        return [True, False]

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @classproperty
    def required(self) -> bool:
        return False


# List of all model-related configuration options
model_options: List[Type[ModelOption]] = [
    ModelChoice,
    OptimizerChoice,
    LearningRate,
    DropoutRate,
    Epsilon,
    LossChoice,
    DisablePytorchCompile
]
