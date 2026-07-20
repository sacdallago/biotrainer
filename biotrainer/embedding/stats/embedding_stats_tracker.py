from __future__ import annotations

import torch
import numpy as np

from typing import Iterable
from biotrainer_core.data_classes import EmbeddingStats


class EmbeddingStatsTracker:
    """ Tracks statistics about embeddings (currently only per-residue embeddings)."""

    def __init__(self, embedder_name: str):
        self._embedder_name = embedder_name

        self.max = float("-inf")
        self.min = float("inf")
        self.dims = None
        self.n_tracked = 0

    def track_entire_dataset(self, per_residue_embeddings: Iterable):
        for emb in per_residue_embeddings:
            self.track(emb)

    def track(self, per_residue_embedding):
        # TODO Use torch
        if isinstance(per_residue_embedding, torch.Tensor):
            per_residue_embedding = per_residue_embedding.cpu().numpy()
        if isinstance(per_residue_embedding, list):
            per_residue_embedding = np.array(per_residue_embedding)

        self.n_tracked += per_residue_embedding.shape[0]
        if self.dims is None:
            self.dims = per_residue_embedding.shape[1]
        self.min = min(self.min, per_residue_embedding.min())
        self.max = max(self.max, per_residue_embedding.max())

    def get_stats(self) -> EmbeddingStats:
        return EmbeddingStats(embedder_name=self._embedder_name,
                              dims=self.dims,
                              n_tracked=self.n_tracked,
                              min=self.min,
                              max=self.max)
