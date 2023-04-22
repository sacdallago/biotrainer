from typing import List, Type, Any, Union

from bio_embeddings.embed import __all__ as bio_embedders

from .config_option import ConfigOption, FileOption
from ..protocols import Protocols


class EmbedderName(FileOption):

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
    def allowed_formats(self) -> List[str]:
        return [".py"]

    def allow_download(self) -> bool:
        return False

    def is_value_valid(self, value: Any) -> bool:
        if ".py" in value:
            return self._validate_file(value)
        else:
            available_embedders = [available_embedder for available_embedder in bio_embedders
                                   if "Interface" not in available_embedder]
            return value in available_embedders


class EmbeddingsFile(FileOption):
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

    def allow_download(self) -> bool:
        return True


embedding_options: List = [EmbedderName, EmbeddingsFile]