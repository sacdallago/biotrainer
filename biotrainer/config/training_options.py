from abc import ABC
from typing import List, Any, Union

from .config_option import ConfigOption, FileOption, classproperty


class TrainingOption(ConfigOption, ABC):

    @classproperty
    def category(self) -> str:
        return "training_option"


class NumberEpochs(TrainingOption, ConfigOption):

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
        return value > 0

    @classproperty
    def required(self) -> bool:
        return False


class BatchSize(TrainingOption, ConfigOption):

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
        return value > 0

    @classproperty
    def required(self) -> bool:
        return False


class Patience(TrainingOption, ConfigOption):

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
        return value > 0

    @classproperty
    def required(self) -> bool:
        return False


class Shuffle(TrainingOption, ConfigOption):

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
        return value > 0 or value == config_option.default_value

    @classproperty
    def required(self) -> bool:
        return False


training_options: List = [NumberEpochs, BatchSize, Patience, Shuffle,
                          UseClassWeights, AutoResume, PretrainedModel, LimitedSampleSize]
