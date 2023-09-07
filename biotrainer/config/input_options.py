from abc import ABC
from pathlib import Path
from typing import List, Type, Any, Union

from .config_option import FileOption, classproperty
from ..protocols import Protocol


class InputOption(FileOption, ABC):

    @classproperty
    def category(self) -> str:
        return "input_option"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False


class SequenceFile(InputOption, FileOption):

    @classproperty
    def name(self) -> str:
        return "sequence_file"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @classproperty
    def allowed_formats(self) -> List[str]:
        return ["fasta"]

    @classproperty
    def allow_download(self) -> bool:
        return True

    @classproperty
    def required(self) -> bool:
        return True


class LabelsFile(InputOption, FileOption):

    @classproperty
    def name(self) -> str:
        return "labels_file"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @classproperty
    def allowed_formats(self) -> List[str]:
        return ["fasta"]

    @classproperty
    def allow_download(self) -> bool:
        return True

    @classproperty
    def allowed_protocols(self) -> List[Protocol]:
        return [Protocol.residue_to_class]

    @classproperty
    def required(self) -> bool:
        return self._protocol in Protocol.per_residue_protocols()


class MaskFile(InputOption, FileOption):
    @classproperty
    def name(self) -> str:
        return "mask_file"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @classproperty
    def allowed_formats(self) -> List[str]:
        return ["fasta"]

    @classproperty
    def allow_download(self) -> bool:
        return True

    @classproperty
    def allowed_protocols(self) -> List[Protocol]:
        return [Protocol.residue_to_class]

    @classproperty
    def required(self) -> bool:
        return False


input_options: List = [SequenceFile, LabelsFile, MaskFile]