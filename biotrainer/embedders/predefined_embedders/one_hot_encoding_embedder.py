import torch
import numpy as np

from ..interfaces import EmbedderInterface

from ...utilities import AMINO_ACIDS

# Create a mapping of amino acids to their index
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class OneHotEncodingEmbedder(EmbedderInterface):
    """
    Baseline embedder: One hot encoding as per-residue embedding, amino acid composition for per-protein

    This embedder is meant to be used as naive baseline for comparing different types of inputs or training methods.
    """

    embedding_dimension = len(AA_TO_INDEX.keys())
    name = "one_hot_encoding"

    def __init__(self):
        self.eye_matrix = np.eye(self.embedding_dimension, dtype=np.float32)

    def _embed_single(self, sequence: str) -> torch.tensor:
        # Convert sequence to indices
        indices = np.fromiter((AA_TO_INDEX.get(aa, -1) for aa in sequence), dtype=np.int8)

        # Use advanced indexing of identity matrix to create one-hot encoding
        return torch.tensor(self.eye_matrix[indices])

    @staticmethod
    def reduce_per_protein(embedding: torch.tensor) -> torch.tensor:
        """This returns the amino acid composition of the sequence as vector"""
        return embedding.mean(axis=0)
