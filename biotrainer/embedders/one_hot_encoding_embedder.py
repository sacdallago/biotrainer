# Changed version from the orignal one from Konstantin Schütze (konstin, https://github.com/konstin) from
# bio_embeddings repository (https://github.com/sacdallago/bio_embeddings)
# Original file: https://github.com/sacdallago/bio_embeddings/blob/efb9801f0de9b9d51d19b741088763a7d2d0c3a2/bio_embeddings/embed/one_hot_encoding_embedder.py

import numpy
from numpy import ndarray

from .embedder_interfaces import EmbedderInterface

AMINO_ACIDS = numpy.asarray(list("ACDEFGHIKLMNPQRSTVWXY"))


class OneHotEncodingEmbedder(EmbedderInterface):
    """Baseline embedder: One hot encoding as per-residue embedding, amino acid composition for per-protein

    This embedder is meant to be used as naive baseline for comparing different types of inputs or training method.

    While option such as device aren't used, you may still pass them for consistency.
    """

    number_of_layers = 1
    embedding_dimension = len(AMINO_ACIDS)
    name = "one_hot_encoding"

    def _embed_single(self, sequence: str) -> ndarray:
        one_hot = [AMINO_ACIDS == i for i in sequence]
        return numpy.stack(one_hot).astype(numpy.float32)

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        """This returns the amino acid composition of the sequence as vector"""
        return embedding.mean(axis=0)
