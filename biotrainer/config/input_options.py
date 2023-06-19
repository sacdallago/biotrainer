from abc import ABC
from pathlib import Path
from typing import List, Type, Any, Union

from .config_option import FileOption
from ..protocols import Protocol


class InputOption(FileOption, ABC):

    @property
    def category(self) -> str:
        return "input_option"


class SequenceFile(InputOption, FileOption):
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
        return [str, Path]

    @property
    def allow_download(self) -> bool:
        return True

    @property
    def required(self) -> bool:
        return True


class LabelsFile(InputOption, FileOption):

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
        return [str, Path]

    @property
    def allow_download(self) -> bool:
        return True

    @property
    def allowed_protocols(self) -> List[Protocol]:
        return [Protocol.residue_to_class]

    @property
    def required(self) -> bool:
        return self._protocol in Protocol.per_residue_protocols()


class MaskFile(InputOption, FileOption):
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
        return [str, Path]

    @property
    def allow_download(self) -> bool:
        return True

    @property
    def allowed_protocols(self) -> List[Protocol]:
        return [Protocol.residue_to_class]

    @property
    def required(self) -> bool:
        return False


input_options: List = [SequenceFile, LabelsFile, MaskFile]