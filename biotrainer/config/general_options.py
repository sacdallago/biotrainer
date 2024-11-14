from abc import ABC
from pathlib import Path
from typing import List, Any, Union, Type

from .config_option import ConfigOption, classproperty
from ..protocols import Protocol


class GeneralOption(ConfigOption, ABC):
    """
    Abstract base class for general configuration options.

    Extends `ConfigOption` to provide a specialized framework for general options
    within a protocol. This class serves as a foundation for specific general options
    by setting common attributes and behaviors.
    """

    @classproperty
    def category(self) -> str:
        return "general_option"


class ProtocolOption(GeneralOption, ConfigOption):
    """
    Configuration option for specifying the protocol to use.

    This option allows users to select a specific protocol from the available
    predefined protocols. It is a required option, ensuring that a valid protocol
    is always specified.
    """

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
    """
    Configuration option for specifying the computing device.

    This option allows users to select the device on which computations will be performed.
    Supported devices include CPU, CUDA (GPU), and MPS (Apple Silicon). It is an optional
    setting with a default value of an empty string, indicating automatic device selection.
    """

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
        return ["cpu", "cuda", "mps"]

    @classproperty
    def allowed_protocols(self) -> List[Protocol]:
        return Protocol.all()

    @classproperty
    def required(self) -> bool:
        return False

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        """
        Validates the provided device value.

        This method checks whether the device value is either a supported device name
        or a CUDA device specified with an index (e.g., "cuda:0").

        Args:
            config_option (ConfigOption): The configuration option instance.
            value (Any): The device value to validate.

        Returns:
            bool: True if the device value is valid, False otherwise.
        """
        if "cuda:" in str(value):
            return str(value).split(":")[-1].isdigit()  # cuda:0, cuda:1 ..
        return value in config_option.possible_values


class Interaction(GeneralOption, ConfigOption):
    """
    Configuration option for specifying the interaction method.

    This option allows users to select the method of interaction, either by multiplying
    or concatenating embeddings. It is an optional setting with a default value of an empty string,
    indicating no specific interaction method is set. Enabling it will interpret the given sequence file as an interaction dataset.
    """

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
    """
    Configuration option for setting the random seed.

    This option allows users to specify a seed value for random number generators to ensure
    reproducibility. It is an optional setting with a default value of 42.
    """

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
    """
    Configuration option for deciding whether to save split identifiers.

    This option allows users to choose whether to save the identifiers of data splits
    (e.g., training, validation, testing) during processing. It is an optional setting
    with a default value of False.
    """

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
    """
    Configuration option for enabling or disabling sanity checks.

    This option allows users to enable or disable sanity checks during processing to
    verify the integrity and correctness of data and configurations. It is an optional
    setting with a default value of True.
    """

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
    """
    Configuration option for ignoring file inconsistencies.

    This option allows users to choose whether to ignore inconsistencies in file-related
    configurations. It is an optional setting with a default value of False.
    """

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
    """
    Configuration option for specifying the output directory.

    This option allows users to define the directory where output files and results
    will be stored. It is an optional setting with a default value of "output". If provided,
    the directory path is converted to an absolute path.
    """

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
        """
        Transforms the output directory path to an absolute path if necessary.

        If a configuration file path is provided, this method converts the relative
        output directory path to an absolute path based on the configuration file location.

        Args:
            config_file_path (Path, optional): Path to the configuration file directory.
        """
        if config_file_path is not None:
            self.value = str((config_file_path / self.value).absolute())

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        return isinstance(value, (str, Path))


# List of all general configuration options
general_options: List[Type[ConfigOption]] = [
    ProtocolOption,
    Device,
    Interaction,
    Seed,
    SaveSplitIds,
    SanityCheck,
    IgnoreFileInconsistencies,
    OutputDirectory
]
