from numpy import ndarray
from typing import Iterable, List

from .embedder_interfaces import EmbedderInterface


class CustomEmbedder(EmbedderInterface):

    # Embedder name, also used to create the directory where the computed embeddings are stored
    name = "custom_embedder"

    @staticmethod
    def _preprocess_sequences(sequences: Iterable[str]) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        raise NotImplementedError

    def _embed_single(self, sequence: str) -> ndarray:
        raise NotImplementedError
