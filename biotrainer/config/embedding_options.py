from abc import ABC
from pathlib import Path
from typing import List, Any, Union, Type

from .config_option import FileOption, classproperty, ConfigOption
from ..embedders import get_predefined_embedder_names


class EmbeddingOption(ConfigOption, ABC):
    """
    Abstract base class for embedding-related configuration options.

    Extends `ConfigOption` to provide a specialized framework for embedding options
    within a protocol. This class serves as a foundation for specific embedding options
    by setting common attributes and behaviors.
    """

    @classproperty
    def category(self) -> str:
        return "embedding_option"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False


class EmbedderName(EmbeddingOption, FileOption):
    """
    Configuration option for specifying the name of the embedder.

    This option allows users to define which embedder to use by specifying either
    a predefined embedder name or a custom embedder script in `.py` format.
    """

    @classproperty
    def name(self) -> str:
        return "embedder_name"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "custom_embeddings"

    @classproperty
    def required(self) -> bool:
        return True

    @classproperty
    def allowed_formats(self) -> List[str]:
        return [".py"]

    @classproperty
    def allow_download(self) -> bool:
        return False

    @staticmethod
    def _is_value_valid(config_option: ConfigOption, value) -> bool:
        """
        Validates the provided embedder name or script path.

        This method checks whether the provided value is either a predefined embedder name,
        the default value, or a valid file path with the ".py" extension.

        Args:
            config_option (ConfigOption): The configuration option instance.
            value (str): The embedder name or file path to validate.

        Returns:
            bool: True if the value is valid, False otherwise.
        """
        if ".py" not in value and "/" not in value:
            return value in get_predefined_embedder_names() or value == config_option.default_value
        return config_option.allow_download if super()._is_url(value) else True

    def transform_value_if_necessary(self, config_file_path: Path = None):
        """
        Transforms the embedder value if it points to a Python script.

        If the embedder name includes a ".py" extension, this method delegates the transformation
        to the superclass to handle any necessary file operations.

        Args:
            config_file_path (Path, optional): Path to the configuration file directory.
        """
        if ".py" in self.value:
            super().transform_value_if_necessary(config_file_path)


class UseHalfPrecision(EmbeddingOption, ConfigOption):
    """
    Configuration option to toggle the use of half-precision (FP16) in embeddings.

    This option allows users to enable or disable half-precision computations, which can
    lead to performance improvements and reduced memory usage at the potential cost of
    numerical precision.
    """

    @classproperty
    def name(self) -> str:
        return "use_half_precision"

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


class EmbeddingsFile(EmbeddingOption, FileOption):
    """
    Configuration option for specifying the embeddings file.

    This option allows users to provide a file containing precomputed embeddings,
    typically in HDF5 (`.h5`) format. It supports downloading from URLs if permitted.
    """

    @classproperty
    def name(self) -> str:
        return "embeddings_file"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @classproperty
    def allowed_formats(self) -> List[str]:
        return ["h5"]

    @classproperty
    def required(self) -> bool:
        return False

    @classproperty
    def allow_download(self) -> bool:
        return True


# List of all embedding-related configuration options
embedding_options: List[Type[EmbeddingOption]] = [EmbedderName, UseHalfPrecision, EmbeddingsFile]
