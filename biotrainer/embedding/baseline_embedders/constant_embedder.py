import torch
import numpy as np

from ..interfaces import BaselineEmbedder


class ConstantEmbedder(BaselineEmbedder):
    """
    Baseline embedder: Generate random, but constant (same regardless of input) 128xL embedding vectors.

    This embedder is meant to be used as a naive baseline to compare against other pretrained embedders.
    """

    embedding_dimension = 128
    name = "constant_embedder"

    def __init__(self):
        self.rng = np.random.default_rng()
        self.embedding = torch.tensor(self.rng.random(self.embedding_dimension, dtype=np.float32))

    def _embed_single(self, sequence: str) -> torch.Tensor:
        return self.embedding.repeat(len(sequence), 1)

    def compute_attention_map(self, sequence: str) -> torch.Tensor:
        """ Constant attention: 1 for every position """
        return torch.ones((len(sequence), len(sequence), 1), dtype=torch.float32)
