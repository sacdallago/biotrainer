from numpy import ndarray
from typing import Iterable, List, Generator, Optional

from .embedder_interfaces import EmbedderInterface


class CustomEmbedder(EmbedderInterface):

    # Embedder name, used to create the directory where the computed embeddings are stored
    name = "custom_embedder"

    def embed_many(
        self, sequences: Iterable[str], batch_size: Optional[int] = None
    ) -> Generator[ndarray, None, None]:
        """
        Method to embed all sequences from the provided iterable.
        This is the function that should be overwritten by most custom embedders, because it allows full control
        over the whole embeddings generation process. Other functions are optional to use and overwrite, except
        reduce_per_protein (if necessary).

        Yields embedding for one sequence at a time.

        :param sequences: List of proteins as AA strings
        :param batch_size: For embedders that profit from batching, this is maximum number of AA per batch

        :return: A list object with embeddings of the sequences.
        """
        raise NotImplementedError

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        """
        Reduces per-residue embeddings to per-protein embeddings by any chosen method (like mean/max).
        Must be overwritten if the embedder should be capable of creating such embeddings.
        """
        raise NotImplementedError

    def _embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        raise NotImplementedError

    def _embed_single(self, sequence: str) -> ndarray:
        raise NotImplementedError


