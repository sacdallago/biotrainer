from abc import ABC
from typing import List, Type, Any, Union

from .config_option import ConfigOption, FileOption
from ..protocols import Protocol


class EmbeddingOption(ConfigOption, ABC):

    @property
    def category(self) -> str:
        return "embedding_option"


class EmbedderName(EmbeddingOption, FileOption):

    @property
    def name(self) -> str:
        return "embedder_name"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "custom_embeddings"

    @property
    def possible_types(self) -> List[Type]:
        return [str]

    @property
    def possible_values(self) -> List[Any]:
        from bio_embeddings.embed import __all__ as bio_embedders
        available_embedders = [available_embedder for available_embedder in bio_embedders
                               if "Interface" not in available_embedder]
        return available_embedders

    @property
    def required(self) -> bool:
        return True

    @property
    def allowed_formats(self) -> List[str]:
        return [".py"]

    @property
    def allow_download(self) -> bool:
        return False

    def is_value_valid(self, value: Any) -> bool:
        if ".py" in value:
            return self._validate_file(value)
        else:
            return value in self.possible_values


class EmbeddingsFile(EmbeddingOption, FileOption):

    @property
    def name(self) -> str:
        return "embeddings_file"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return ""

    @property
    def allowed_formats(self) -> List[str]:
        return ["h5"]

    @property
    def possible_types(self) -> List[Type]:
        return [str]

    @property
    def required(self) -> bool:
        return False

    @property
    def allow_download(self) -> bool:
        return True


embedding_options: List = [EmbedderName, EmbeddingsFile]