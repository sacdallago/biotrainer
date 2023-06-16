from abc import ABC
from typing import List, Type, Any, Union

from .config_option import ConfigOption, FileOption
from ..protocols import Protocol


class TrainingOption(ConfigOption, ABC):

    @property
    def category(self) -> str:
        return "training_option"


class NumberEpochs(TrainingOption, ConfigOption):
    @property
    def name(self) -> str:
        return "num_epochs"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 200

    @property
    def possible_types(self) -> List[Type]:
        return [int]

    def is_value_valid(self, value: Any) -> bool:
        return value > 0

    @property
    def required(self) -> bool:
        return False


class BatchSize(TrainingOption, ConfigOption):

    @property
    def name(self) -> str:
        return "batch_size"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 128

    @property
    def possible_types(self) -> List[Type]:
        return [int]

    def is_value_valid(self, value: Any) -> bool:
        return value > 0

    @property
    def required(self) -> bool:
        return False


class Patience(TrainingOption, ConfigOption):

    @property
    def name(self) -> str:
        return "patience"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 10

    @property
    def possible_types(self) -> List[Type]:
        return [int]

    def is_value_valid(self, value: Any) -> bool:
        return value > 0

    @property
    def required(self) -> bool:
        return False


class Shuffle(TrainingOption, ConfigOption):
    @property
    def name(self) -> str:
        return "shuffle"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return True

    @property
    def possible_types(self) -> List[Type]:
        return [bool]

    @property
    def possible_values(self) -> List[Any]:
        return [True, False]

    @property
    def required(self) -> bool:
        return False


class UseClassWeights(TrainingOption, ConfigOption):
    @property
    def name(self) -> str:
        return "use_class_weights"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return False

    @property
    def possible_types(self) -> List[Type]:
        return [bool]

    @property
    def possible_values(self) -> List[Any]:
        return [True, False]

    @property
    def required(self) -> bool:
        return False


class AutoResume(TrainingOption, ConfigOption):
    @property
    def name(self) -> str:
        return "auto_resume"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return False

    @property
    def possible_types(self) -> List[Type]:
        return [bool]

    @property
    def possible_values(self) -> List[Any]:
        return [True, False]

    @property
    def required(self) -> bool:
        return False


class PretrainedModel(TrainingOption, FileOption):
    @property
    def name(self) -> str:
        return "pretrained_model"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @property
    def allowed_formats(self) -> List[str]:
        return [".pt"]

    @property
    def possible_types(self) -> List[Type]:
        return [str]

    @property
    def allow_download(self) -> bool:
        return False

    @property
    def required(self) -> bool:
        return False


class LimitedSampleSize(TrainingOption, ConfigOption):
    @property
    def name(self) -> str:
        return "limited_sample_size"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return -1

    @property
    def possible_types(self) -> List[Type]:
        return [int]

    def is_value_valid(self, value: Any) -> bool:
        return value > 0

    @property
    def required(self) -> bool:
        return False


training_options: List = [NumberEpochs, BatchSize, Patience, Shuffle,
                          UseClassWeights, AutoResume, PretrainedModel, LimitedSampleSize]
