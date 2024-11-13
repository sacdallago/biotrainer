from abc import ABC
from typing import List, Any, Union, Type

from .config_option import FileOption, classproperty
from ..protocols import Protocol


class InputOption(FileOption, ABC):
    """
    Abstract base class for input-related configuration options.

    Extends `FileOption` to provide a specialized framework for input options
    within a protocol. This class serves as a foundation for specific input options
    by setting common attributes and behaviors.
    """

    @classproperty
    def category(self) -> str:
        return "input_option"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False


class SequenceFile(InputOption, FileOption):
    """
    Configuration option for specifying the sequence file.

    This option allows users to provide a sequence file, typically in FASTA format,
    which contains the biological sequences to be used for training. It supports downloading
    from URLs if permitted and ensures that the file adheres to the allowed formats.
    """

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
    """
    Configuration option for specifying the labels file.

    This option allows users to provide a labels file, typically in FASTA format,
    which contains the labels corresponding to the biological sequences. It supports
    downloading from URLs if permitted and is conditionally required based on the selected protocol.
    """

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
    """
    Configuration option for specifying the mask file.

    This option allows users to provide a mask file, typically in FASTA format,
    which contains masks corresponding to the biological sequences. It supports
    downloading from URLs if permitted and is optional based on the selected protocol.
    """

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


# List of all input-related configuration options
input_options: List[Type[InputOption]] = [SequenceFile, LabelsFile, MaskFile]
