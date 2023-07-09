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
    def possible_types(self) -> List[Type]:
        return [type(Protocol)]

    @property
    def possible_values(self) -> List[Any]:
        return Protocol.all()

    @classproperty
    def required(self) -> bool:
        return True

    def is_value_valid(self) -> bool:
        return Protocol.__getitem__(self.value) is not None


class Interaction(GeneralOption, ConfigOption):

    @classproperty
    def name(self) -> str:
        return "interaction"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @classproperty
    def possible_types(self) -> List[Type]:
        return [str]

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
    def possible_types(self) -> List[Type]:
        return [int]

    def is_value_valid(self) -> bool:
        # For range see: https://pytorch.org/docs/stable/generated/torch.manual_seed.html#torch.manual_seed
        return -0x8000_0000_0000_0000 < self.value < 0xffff_ffff_ffff_ffff

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
    def possible_types(self) -> List[Type]:
        return [bool]

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
    def possible_types(self) -> List[Type]:
        return [bool]

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
    def possible_types(self) -> List[Type]:
        return [bool]

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
    def possible_types(self) -> List[Type]:
        return [str, Path]

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

    def is_value_valid(self) -> bool:
        return type(self.value) in self.possible_types


general_options: List = [ProtocolOption, Interaction, Seed, SaveSplitIds, SanityCheck, IgnoreFileInconsistencies,
                         OutputDirectory]
