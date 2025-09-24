import torch
import numpy as np

from ..interfaces import EmbedderInterface

from ...utilities import AMINO_ACIDS

class AAOntologyEmbedder(EmbedderInterface):
    """
    Baseline embedder: Uses plain scales from AAOntology.

    AAOntology provides amino-acid associated feature scales: https://doi.org/10.1016/j.jmb.2024.168717
    """

    name = "AAOntology"

    def __init__(self):
        # Create efficient lookup structures
        self._setup_efficient_mapping()

    def _setup_efficient_mapping(self):
        """Setup efficient amino acid to scale mapping"""
        import aaanalysis as aa

        scales = aa.load_scales()

        # Define amino acid order (including X as placeholder)
        self.embedding_dimension = len(scales.columns)  # Number of scales (586)

        # Create mapping from amino acid to index
        self.aa_to_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

        # Create the lookup matrix: shape (21, 586) for 20 AAs + X
        self.lookup_matrix = np.zeros((len(AMINO_ACIDS), self.embedding_dimension), dtype=np.float32)

        # Fill the lookup matrix with scale values
        for i, aa in enumerate(AMINO_ACIDS):
            # Exclude X - only zeros
            if aa == "X":
                continue
            if aa in scales.index:
                self.lookup_matrix[i] = scales.loc[aa].values.astype(np.float32)

    def _embed_single(self, sequence: str) -> torch.tensor:
        """Convert sequence to scale-based embedding"""
        indices = np.fromiter(
            (self.aa_to_index.get(aa, self.aa_to_index['X']) for aa in sequence),
            dtype=np.int32
        )

        # Use advanced indexing to get scale values for each position
        return torch.tensor(self.lookup_matrix[indices])

    @staticmethod
    def reduce_per_protein(embedding: torch.tensor) -> torch.tensor:
        """Returns the mean of all scale values across the sequence"""
        return embedding.mean(axis=0)
