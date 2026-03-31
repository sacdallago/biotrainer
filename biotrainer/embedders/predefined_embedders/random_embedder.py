import torch
import numpy as np

from ..interfaces import BaselineEmbedder


class RandomEmbedder(BaselineEmbedder):
    """
    Baseline embedder: Generate random 128xL embedding vectors.

    This embedder is meant to be used as a naive baseline to compare against other pretrained embedders.
    """

    embedding_dimension = 128
    name = "random_embedder"

    def __init__(self):
        self.rng = np.random.default_rng()

    def _embed_single(self, sequence: str) -> torch.Tensor:
        return torch.tensor(self.rng.random((len(sequence), self.embedding_dimension), dtype=np.float32))
