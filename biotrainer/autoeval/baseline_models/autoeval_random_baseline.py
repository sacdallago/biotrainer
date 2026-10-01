import torch

from typing import Generator, List, Tuple

from .baseline_interface import AutoEvalBaseline
from ...bioengineer import BioEngineer, BioEngineerBaseline
from ...embedding import CustomEmbedder, RandomEmbedder


class _CustomRandomEmbedder(CustomEmbedder):
    def __init__(self):
        self._embedder = RandomEmbedder()

    def per_residue(self, sequences: List[str]) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """ Embed a list of sequences and yield a tuple of (sequence, per-residue embedding for this sequence) """
        for seq in sequences:
            yield seq, self._embedder._embed_single(seq)

    def per_sequence(self, sequences: List[str]) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """ Embed a list of sequences and yield a tuple of (sequence, per-sequence embedding for this sequence) """
        for seq in sequences:
            res_embedding = self._embedder._embed_single(seq)
            yield seq, self._embedder.reduce_per_protein(res_embedding)

    def compute_attention_map(self, sequence: str) -> torch.Tensor:
        """ Compute the attention map for the given sequence (if model has attention) - used for supervised contact prediction """
        return self._embedder.compute_attention_map(sequence)


class AutoEvalRandomBaseline(AutoEvalBaseline):
    name: str = "random_baseline"

    def embedder(self) -> CustomEmbedder:
        return _CustomRandomEmbedder()

    def bioengineer(self) -> BioEngineer:
        return BioEngineer.from_baseline(baseline=BioEngineerBaseline.RANDOM_BASELINE)