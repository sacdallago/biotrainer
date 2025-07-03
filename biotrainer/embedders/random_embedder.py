import numpy as np

from numpy import ndarray

from .embedder_interfaces import EmbedderInterface


class RandomEmbedder(EmbedderInterface):
    """
    Baseline embedder: Generate random 128xL embedding vectors.

    This embedder is meant to be used as a naive baseline to compare against other pretrained embedders.
    """

    embedding_dimension = 128
    name = "random_embedder"

    def __init__(self):
        self.rng = np.random.default_rng()

    def _embed_single(self, sequence: str) -> ndarray:
        return self.rng.random((len(sequence), self.embedding_dimension), dtype=np.float32)

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        """Reduce via mean"""
        return embedding.mean(axis=0)
