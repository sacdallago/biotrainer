import os.path
from abc import ABC
from pathlib import Path
from typing import List, Type, Any, Union

from .config_option import ConfigOption
from ..protocols import Protocol


class GeneralOption(ConfigOption, ABC):

    @property
    def category(self) -> str:
        return "general_option"


class ProtocolOption(GeneralOption, ConfigOption):

    @property
    def name(self) -> str:
        return "protocol"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @property
    def possible_types(self) -> List[Type]:
        return [type(Protocol)]

    @property
    def possible_values(self) -> List[Any]:
        return Protocol.all()

    @property
    def required(self) -> bool:
        return True

    def is_value_valid(self, value: Any) -> bool:
        return Protocol.__getitem__(value) is not None


class Interaction(GeneralOption, ConfigOption):

    @property
    def name(self) -> str:
        return "interaction"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @property
    def possible_types(self) -> List[Type]:
        return [str]

    @property
    def possible_values(self) -> List[Any]:
        return ["multiply", "concat"]

    @property
    def allowed_protocols(self) -> List[Protocol]:
        return [Protocol.sequence_to_class, Protocol.sequence_to_value]

    @property
    def required(self) -> bool:
        return False


class Seed(GeneralOption, ConfigOption):

    @property
    def name(self) -> str:
        return "seed"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return 42

    @property
    def possible_types(self) -> List[Type]:
        return [int]

    def is_value_valid(self, value: Any) -> bool:
        # For range see: https://pytorch.org/docs/stable/generated/torch.manual_seed.html#torch.manual_seed
        return -0x8000_0000_0000_0000 < value < 0xffff_ffff_ffff_ffff

    @property
    def required(self) -> bool:
        return False


class SaveSplitIds(GeneralOption, ConfigOption):
    @property
    def name(self) -> str:
        return "save_split_ids"

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


class SanityCheck(GeneralOption, ConfigOption):
    @property
    def name(self) -> str:
        return "sanity_check"

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


class IgnoreFileInconsistencies(GeneralOption, ConfigOption):
    @property
    def name(self) -> str:
        return "ignore_file_inconsistencies"

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


class OutputDirectory(GeneralOption, ConfigOption):

    @property
    def name(self) -> str:
        return "output_dir"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "output"

    @property
    def possible_types(self) -> List[Type]:
        return [str, Path]

    @property
    def possible_values(self) -> List[Any]:
        return []

    @property
    def allowed_protocols(self) -> List[Protocol]:
        return Protocol.all()

    @property
    def required(self) -> bool:
        return False

    def is_value_valid(self, value: Any) -> bool:
        return os.path.exists(str(value))


general_options: List = [ProtocolOption, Interaction, Seed, SaveSplitIds, SanityCheck, IgnoreFileInconsistencies,
                         OutputDirectory]
