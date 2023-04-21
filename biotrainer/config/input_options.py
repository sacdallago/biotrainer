from typing import List, Type, Any, Union

from .config_option import ConfigOption, FileOption
from .protocols import Protocols


class SequenceFile(FileOption):
    @property
    def name(self) -> str:
        return "sequence_file"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @property
    def allowed_formats(self) -> List[str]:
        return ["fasta"]

    @property
    def possible_types(self) -> List[Type]:
        return [str]

    def allow_download(self) -> bool:
        return True


class LabelsFile(FileOption):

    @property
    def name(self) -> str:
        return "labels_file"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @property
    def allowed_formats(self) -> List[str]:
        return ["fasta"]

    @property
    def possible_types(self) -> List[Type]:
        return [str]

    def allow_download(self) -> bool:
        return True

    def allowed_protocols(self) -> List[Protocols]:
        return [Protocols.residue_to_class]


class MaskFile(FileOption):
    @property
    def name(self) -> str:
        return "mask_file"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @property
    def allowed_formats(self) -> List[str]:
        return ["fasta"]

    @property
    def possible_types(self) -> List[Type]:
        return [str]

    def allow_download(self) -> bool:
        return True

    def allowed_protocols(self) -> List[Protocols]:
        return [Protocols.residue_to_class]