from typing import List, Type, Any, Union

from .config_option import ConfigOption
from ..protocols import Protocols


class ProtocolOption(ConfigOption):

    @property
    def name(self) -> str:
        return "protocol"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return None

    @property
    def possible_types(self) -> List[Type]:
        return [type(Protocols)]

    def is_value_valid(self, value: Any) -> bool:
        return value in Protocols.all()


class Interaction(ConfigOption):

    @property
    def name(self) -> str:
        return "interaction"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @property
    def possible_types(self) -> List[Type]:
        return [str]

    def is_value_valid(self, value: Any) -> bool:
        return value in ["multiply", "concat"]

    def allowed_protocols(self) -> List[Protocols]:
        return [Protocols.sequence_to_class, Protocols.sequence_to_value]


class Seed(ConfigOption):

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


class SaveSplitIds(ConfigOption):
    @property
    def name(self) -> str:
        return "save_split_ids"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return False

    @property
    def possible_types(self) -> List[Type]:
        return [bool]

    def is_value_valid(self, value: Any) -> bool:
        return value in [True, False]


class SanityCheck(ConfigOption):
    @property
    def name(self) -> str:
        return "sanity_check"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return True

    @property
    def possible_types(self) -> List[Type]:
        return [bool]

    def is_value_valid(self, value: Any) -> bool:
        return value in [True, False]


class IgnoreFileInconsistencies(ConfigOption):
    @property
    def name(self) -> str:
        return "ignore_file_inconsistencies"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return False

    @property
    def possible_types(self) -> List[Type]:
        return [bool]

    def is_value_valid(self, value: Any) -> bool:
        return value in [True, False]


general_options: List = [ProtocolOption, Interaction, Seed, SaveSplitIds, SanityCheck, IgnoreFileInconsistencies]