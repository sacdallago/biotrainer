from typing import Optional

import hashlib

import numpy as np
import torch

from ..interfaces import BaselineEmbedder


class RandomEmbedder(BaselineEmbedder):
    """
    Baseline embedder: Generate random 128xL embedding vectors.

    This embedder is meant to be used as a naive baseline to compare against other pretrained embedders.

    With a seed, a sequence's vectors depend only on the seed and the sequence itself, not on what else
    is embedded alongside it or in which order.
    """

    embedding_dimension = 128
    name = "random_embedder"

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.default_rng()

    def _rng_for(self, sequence: str) -> np.random.Generator:
        if self.seed is None:
            return self.rng
        sequence_hash = int.from_bytes(
            hashlib.sha256(sequence.encode()).digest(), "big"
        )
        return np.random.default_rng([self.seed, sequence_hash])

    def _embed_single(self, sequence: str) -> torch.Tensor:
        return torch.tensor(
            self._rng_for(sequence).random(
                (len(sequence), self.embedding_dimension), dtype=np.float32
            )
        )

    def compute_attention_map(self, sequence: str) -> torch.Tensor:
        return torch.tensor(self.rng.random((len(sequence), len(sequence), 1), dtype=np.float32))
