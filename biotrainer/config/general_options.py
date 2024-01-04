import os.path
from abc import ABC
from pathlib import Path
from typing import List, Type, Any, Union

from .config_option import ConfigOption, classproperty
from ..protocols import Protocol


class GeneralOption(ConfigOption, ABC):

    @classproperty
    def category(self) -> str:
        return "general_option"


class ProtocolOption(GeneralOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "protocol"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @property
    def possible_values(self) -> List[Any]:
        return Protocol.all()

    @classproperty
    def required(self) -> bool:
        return True

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return Protocol.__getitem__(value) is not None


class Device(GeneralOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "device"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @property
    def possible_values(self) -> List[Any]:
        return ["cpu", "cuda"]

    @classproperty
    def allowed_protocols(self) -> List[Protocol]:
        return Protocol.all()

    @classproperty
    def required(self) -> bool:
        return False


class Interaction(GeneralOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "interaction"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @property
    def possible_values(self) -> List[Any]:
        return ["multiply", "concat"]

    @classproperty
    def allowed_protocols(self) -> List[Protocol]:
        return [Protocol.sequence_to_class, Protocol.sequence_to_value]

    @classproperty
    def required(self) -> bool:
        return False


class Seed(GeneralOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "seed"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 42

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        # For range see: https://pytorch.org/docs/stable/generated/torch.manual_seed.html#torch.manual_seed
        return -0x8000_0000_0000_0000 < value < 0xffff_ffff_ffff_ffff

    @classproperty
    def required(self) -> bool:
        return False


class SaveSplitIds(GeneralOption, ConfigOption):
    @classproperty
    def name(self) -> str:
        return "save_split_ids"

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


class SanityCheck(GeneralOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "sanity_check"

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


class IgnoreFileInconsistencies(GeneralOption, ConfigOption):
    @classproperty
    def name(self) -> str:
        return "ignore_file_inconsistencies"

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


class OutputDirectory(GeneralOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "output_dir"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "output"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @property
    def possible_values(self) -> List[Any]:
        return []

    @classproperty
    def allowed_protocols(self) -> List[Protocol]:
        return Protocol.all()

    @classproperty
    def required(self) -> bool:
        return False

    def transform_value_if_necessary(self, config_file_path: Path = None):
        self.value = str((config_file_path / self.value).absolute())

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return type(value) in [str, Path]


general_options: List = [ProtocolOption, Device, Interaction,
                         Seed, SaveSplitIds, SanityCheck, IgnoreFileInconsistencies, OutputDirectory]
