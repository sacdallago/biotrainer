from abc import ABC
from pathlib import Path
from typing import List, Any, Union

from .config_option import FileOption, classproperty, ConfigOption

from ..embedders import get_predefined_embedder_names


class EmbeddingOption(ConfigOption, ABC):

    @classproperty
    def category(self) -> str:
        return "embedding_option"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False


class EmbedderName(EmbeddingOption, FileOption):

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
        if ".py" not in value and "/" not in value:
            return value in get_predefined_embedder_names() or value == config_option.default_value
        return config_option.allow_download if super()._is_url(value) else True

    def transform_value_if_necessary(self, config_file_path: Path = None):
        if ".py" in self.value:
            super().transform_value_if_necessary(config_file_path)


class UseHalfPrecision(EmbeddingOption, ConfigOption):

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


embedding_options: List = [EmbedderName, UseHalfPrecision, EmbeddingsFile]
