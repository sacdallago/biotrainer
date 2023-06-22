from abc import ABC
from typing import List, Type, Any, Union

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
    def possible_types(self) -> List[Type]:
        return [int]

    def is_value_valid(self) -> bool:
        return self.value > 0

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
    def possible_types(self) -> List[Type]:
        return [int]

    def is_value_valid(self) -> bool:
        return self.value > 0

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
    def possible_types(self) -> List[Type]:
        return [int]

    def is_value_valid(self) -> bool:
        return self.value > 0

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
    def possible_types(self) -> List[Type]:
        return [bool]

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
    def possible_types(self) -> List[Type]:
        return [bool]

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
    def possible_types(self) -> List[Type]:
        return [bool]

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
    def possible_types(self) -> List[Type]:
        return [str]

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
    def possible_types(self) -> List[Type]:
        return [int]

    def is_value_valid(self) -> bool:
        return self.value > 0 or self.value == self.default_value

    @classproperty
    def required(self) -> bool:
        return False


training_options: List = [NumberEpochs, BatchSize, Patience, Shuffle,
                          UseClassWeights, AutoResume, PretrainedModel, LimitedSampleSize]
