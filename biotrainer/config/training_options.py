from abc import ABC
from typing import List, Any, Union, Type

from .config_option import ConfigOption, FileOption, classproperty


class TrainingOption(ConfigOption, ABC):
    """
    Abstract base class for training-related configuration options.

    Extends `ConfigOption` to provide a specialized framework for training options
    within a protocol. This class serves as a foundation for specific training options
    by setting common attributes and behaviors.
    """

    @classproperty
    def category(self) -> str:
        return "training_option"


class NumberEpochs(TrainingOption, ConfigOption):
    """
    Configuration option for specifying the number of training epochs.

    This option allows users to define one or more epochs for training the model.
    It ensures that the provided number of epochs is a positive integer.
    """

    @classproperty
    def name(self) -> str:
        return "num_epochs"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 200

    @classproperty
    def allow_multiple_values(self) -> bool:
        return True

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return isinstance(value, int) and value > 0

    @classproperty
    def required(self) -> bool:
        return False


class BatchSize(TrainingOption, ConfigOption):
    """
    Configuration option for specifying the batch size.

    This option allows users to define one or more batch sizes for training the model.
    It ensures that the provided batch size is a positive integer.
    """

    @classproperty
    def name(self) -> str:
        return "batch_size"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 128

    @classproperty
    def allow_multiple_values(self) -> bool:
        return True

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return isinstance(value, int) and value > 0

    @classproperty
    def required(self) -> bool:
        return False


class Patience(TrainingOption, ConfigOption):
    """
    Configuration option for specifying patience in early stopping.

    This option allows users to define one or more patience values for early stopping during training.
    It ensures that the provided patience values are positive integers.
    """

    @classproperty
    def name(self) -> str:
        return "patience"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 10

    @classproperty
    def allow_multiple_values(self) -> bool:
        return True

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return isinstance(value, int) and value > 0

    @classproperty
    def required(self) -> bool:
        return False


class Shuffle(TrainingOption, ConfigOption):
    """
    Configuration option for enabling or disabling data shuffling.

    This option allows users to choose whether to shuffle the training data before each epoch.
    It ensures that the provided value is either True or False.
    """

    @classproperty
    def name(self) -> str:
        return "shuffle"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return True

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @property
    def possible_values(self) -> List[Any]:
        return [True, False]

    @classproperty
    def required(self) -> bool:
        return False


class UseClassWeights(TrainingOption, ConfigOption):
    """
    Configuration option for enabling or disabling the use of class weights.

    This option allows users to choose whether to apply class weights during training,
    which can be useful for handling imbalanced datasets. It ensures that the provided
    value is either True or False.
    """

    @classproperty
    def name(self) -> str:
        return "use_class_weights"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return False

    @classproperty
    def allow_multiple_values(self) -> bool:
        return True

    @property
    def possible_values(self) -> List[Any]:
        return [True, False]

    @classproperty
    def required(self) -> bool:
        return False


class AutoResume(TrainingOption, ConfigOption):
    """
    Configuration option for enabling or disabling auto-resuming of training.

    This option allows users to choose whether to automatically resume training from
    the last checkpoint in case of interruptions. It ensures that the provided value
    is either True or False.
    """

    @classproperty
    def name(self) -> str:
        return "auto_resume"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return False

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @property
    def possible_values(self) -> List[Any]:
        return [True, False]

    @classproperty
    def required(self) -> bool:
        return False


class PretrainedModel(TrainingOption, FileOption):
    """
    Configuration option for specifying a pretrained model.

    This option allows users to provide a pretrained model file, typically in PyTorch (.pt) format,
    to initialize the model weights before training. It supports downloading from URLs if permitted.
    """

    @classproperty
    def name(self) -> str:
        return "pretrained_model"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @classproperty
    def allowed_formats(self) -> List[str]:
        return [".pt"]

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @classproperty
    def allow_download(self) -> bool:
        return False

    @classproperty
    def required(self) -> bool:
        return False


class LimitedSampleSize(TrainingOption, ConfigOption):
    """
    Configuration option for limiting the sample size during training.

    This option allows users to define a maximum number of samples to be used for training.
    A value of -1 indicates no limit. It ensures that the provided sample size is either positive
    or set to the default value.
    """

    @classproperty
    def name(self) -> str:
        return "limited_sample_size"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return -1

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return (isinstance(value, int) and value > 0) or value == config_option.default_value

    @classproperty
    def required(self) -> bool:
        return False


# List of all training-related configuration options
training_options: List[Type[TrainingOption]] = [
    NumberEpochs,
    BatchSize,
    Patience,
    Shuffle,
    UseClassWeights,
    AutoResume,
    PretrainedModel,
    LimitedSampleSize
]
