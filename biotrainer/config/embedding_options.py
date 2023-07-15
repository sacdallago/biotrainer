import inspect

from abc import ABC
from pathlib import Path
from typing import List, Type, Any, Union

from .config_option import FileOption, classproperty


class EmbeddingOption(FileOption, ABC):

    @classproperty
    def category(self) -> str:
        return "embedding_option"


class EmbedderName(EmbeddingOption, FileOption):

    @classproperty
    def name(self) -> str:
        return "embedder_name"

    @property
    def default_value(self) -> Union[str, int, float, bool, Any]:
        return "custom_embeddings"

    @classproperty
    def possible_types(self) -> List[Type]:
        return [str]

    @property
    def possible_values(self) -> List[Any]:
        try:
            from bio_embeddings.embed import __all__ as bio_embedders
            available_embedders = [available_embedder for available_embedder in bio_embedders
                                   if "Interface" not in available_embedder]
            return available_embedders
        except ImportError:
            return []

    @classproperty
    def required(self) -> bool:
        return True

    @classproperty
    def allowed_formats(self) -> List[str]:
        return [".py"]

    @classproperty
    def allow_download(self) -> bool:
        return False

    def is_value_valid(self) -> bool:
        if ".py" not in self.value:
            import bio_embeddings
            # Also allow name of embedders instead of class names
            # (one_hot_encoding: name, OneHotEncodingEmbedder: class name)
            all_embedders = [embedder[1].name for embedder in
                             inspect.getmembers(bio_embeddings.embed, inspect.isclass)
                             if "Interface" not in embedder[0]]
            return self.value in self.possible_values or self.value in all_embedders or self.value == self.default_value
        else:
            return super().is_value_valid()

    def transform_value_if_necessary(self, config_file_path: Path = None):
        # Convert class name to bio_embeddings name
        if ".py" not in self.value:
            import bio_embeddings
            all_embedders_dict = {embedder[0]: embedder[1].name for embedder in
                                  inspect.getmembers(bio_embeddings.embed, inspect.isclass)
                                  if "Interface" not in embedder[0]}
            if self.value in all_embedders_dict.keys():
                self.value = all_embedders_dict[self.value]


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
    def possible_types(self) -> List[Type]:
        return [str]

    @classproperty
    def required(self) -> bool:
        return False

    @classproperty
    def allow_download(self) -> bool:
        return True


embedding_options: List = [EmbedderName, EmbeddingsFile]
