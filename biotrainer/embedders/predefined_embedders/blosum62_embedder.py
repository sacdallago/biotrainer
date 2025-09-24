import torch
import numpy as np
import blosum as bl

from ..interfaces import EmbedderInterface

from ...utilities import AMINO_ACIDS


class Blosum62Embedder(EmbedderInterface):
    """
    Baseline embedder: BLOSUM62 matrix as per-residue embedding

    Each amino acid is represented by its corresponding row in the BLOSUM62 substitution matrix.
    This provides evolutionary information about amino acid substitution patterns.
    """

    name = "blosum62"

    def __init__(self):
        # Create efficient lookup structures
        self._setup_efficient_mapping()

    def _setup_efficient_mapping(self):
        """Setup efficient amino acid to BLOSUM score mapping"""
        # Load BLOSUM62 matrix
        matrix = bl.BLOSUM(62)

        self.embedding_dimension = len(AMINO_ACIDS)  # 21 amino acids including X

        # Create mapping from amino acid to index
        self.aa_to_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

        # Create the lookup matrix: shape (21, 21) for all AAs including X
        self.lookup_matrix = np.zeros((len(AMINO_ACIDS), self.embedding_dimension), dtype=np.float32)

        # Fill the lookup matrix with BLOSUM values
        for i, aa in enumerate(AMINO_ACIDS):
            # Get the BLOSUM row for this amino acid in the correct order
            blosum_row = [matrix[aa][target_aa] for target_aa in AMINO_ACIDS]
            self.lookup_matrix[i] = np.array(blosum_row, dtype=np.float32)

    def _embed_single(self, sequence: str) -> torch.tensor:
        """Convert sequence to BLOSUM-based embedding"""
        indices = np.fromiter(
            (self.aa_to_index.get(aa, self.aa_to_index['X']) for aa in sequence),
            dtype=np.int32
        )

        # Use advanced indexing to get BLOSUM rows for each position
        return torch.tensor(self.lookup_matrix[indices])

    @staticmethod
    def reduce_per_protein(embedding: torch.tensor) -> torch.tensor:
        """Returns the average BLOSUM profile of the sequence"""
        return embedding.mean(axis=0)
