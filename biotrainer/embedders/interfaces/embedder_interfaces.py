"""
Abstract interface for Embedder.
File adapted from bio_embeddings (https://github.com/sacdallago/bio_embeddings/blob/efb9801f0de9b9d51d19b741088763a7d2d0c3a2/bio_embeddings/embed/embedder_interfaces.py)
Original Authors:
  Christian Dallago (sacdallago)
  Konstantin Schuetze (konstin)
"""

import abc
import torch

from numpy import ndarray
from typing import List, Generator, Optional, Iterable, Any, Union, Callable

from .preprocessing_strategies import preprocess_sequences_without_whitespaces

from ...utilities import get_logger, get_device_memory

logger = get_logger(__name__)


class EmbedderInterface(abc.ABC):
    name: str
    _model: Optional[Any] = None
    _device: Union[None, str, torch.device] = None
    _preprocessing_strategy: Callable = lambda self, sequences, mask_token: preprocess_sequences_without_whitespaces(sequences, mask_token)

    @abc.abstractmethod
    def _embed_single(self, sequence: str) -> ndarray:
        """
        Returns embedding for one sequence.

        :param sequence: Valid amino acid sequence as String
        :return: An embedding of the sequence.
        """

        raise NotImplementedError


    def get_mask_token(self) -> Optional[str]:
        return None

    def _preprocess_sequences(self, sequences: Iterable[str]) -> List[str]:
        return self._preprocessing_strategy(sequences, self.get_mask_token())

    def _embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        """Computes the embeddings from all sequences in the batch

        The provided implementation is dummy implementation that should be
        overwritten with the appropriate batching method for the model."""
        for sequence in batch:
            yield self._embed_single(sequence)

    def estimate_batch_size(self, preprocessed_sequences: List[str]) -> int:
        memory_gb = get_device_memory(self._device)
        safety_factor = 0.8
        residues_per_gb = 1024  # Estimating 1MB/residue
        batch_size = int(memory_gb * residues_per_gb * safety_factor)

        # Account for separators that are processed by the tokenizer (do not count towards batch_size => double it)
        first_sequence = preprocessed_sequences[0]
        if any([separator in first_sequence for separator in (' ', ',', ';')]):
            batch_size *= 2

        return batch_size

    def embed_many(
            self, sequences: Iterable[str], batch_size: Optional[int] = None
    ) -> Generator[ndarray, None, None]:
        """
        Yields embedding for one sequence at a time.

        :param sequences: List of proteins as AA strings
        :param batch_size: For embedders that profit from batching, this is maximum number of AA per batch
        :return: A list object with embeddings of the sequences.
        """
        sequences = self._preprocess_sequences(sequences)

        if batch_size is None:
            batch_size = self.estimate_batch_size(preprocessed_sequences=sequences)

        if batch_size and batch_size > 1:
            batch = []
            length = 0
            for sequence in sequences:
                if len(sequence) > batch_size:
                    logger.warning(
                        f"A sequence is {len(sequence)} residues long, "
                        f"which is longer than your `batch_size` parameter which is {batch_size}"
                    )
                    yield from self._embed_batch([sequence])
                    continue
                if length + len(sequence) >= batch_size:
                    yield from self._embed_batch(batch)
                    batch = []
                    length = 0
                batch.append(sequence)
                length += len(sequence)
            yield from self._embed_batch(batch)
        else:
            for seq in sequences:
                yield self._embed_single(seq)

    @staticmethod
    def reduce_per_protein(embedding: torch.tensor) -> torch.tensor:
        """
        For a variable size embedding, returns a fixed size embedding encoding all information of a sequence.

        :param embedding: the embedding
        :return: A fixed size embedding (a vector of size N, where N is fixed)
        """
        return embedding.mean(axis=0)


class EmbedderWithFallback(EmbedderInterface, abc.ABC):
    """ Batching embedder that will fallback to the CPU if the embedding on the GPU failed """

    _model: Any

    @abc.abstractmethod
    def _embed_batch_implementation(
            self, batch: List[str], model: Any
    ) -> Generator[torch.tensor, None, None]:
        ...

    @abc.abstractmethod
    def _get_fallback_model(self):
        """Returns a (cached) cpu model.

        Note that the fallback models generally don't support half precision mode and therefore ignore
        the `half_precision_model` option (https://github.com/huggingface/transformers/issues/11546).
        """
        ...

    def _embed_batch(self, batch: List[str]) -> Generator[torch.tensor, None, None]:
        """Tries to get the embeddings in this order:
          * Full batch GPU
          * Single Sequence GPU
          * Single Sequence CPU

        Single sequence processing is done in case of runtime error due to
        a) very long sequence or b) too large batch size
        If this fails, you might want to consider lowering batch_size and/or
        cutting very long sequences into smaller chunks

        Returns unprocessed embeddings
        """
        # No point in having a fallback model when the normal model is CPU already
        if self._device.type == "cpu":
            yield from self._embed_batch_implementation(batch, self._model)
            return

        try:
            yield from self._embed_batch_implementation(batch, self._model.to(self._device))
        except RuntimeError as e:
            if len(batch) == 1:
                logger.warning(
                    f"RuntimeError for sequence with {len(batch[0])} residues - falling back to CPU.. "
                    f"The error most likely means that you don't have enough "
                    f"GPU RAM to embed a protein this long on the GPU. Full error: {e}"
                )
                yield from self._embed_batch_implementation(batch, self._get_fallback_model())
            else:
                logger.warning(
                    f"RuntimeError processing batch of {len(batch)} sequences. "
                    f"You might want to consider adjusting the `batch_size` parameter. "
                    f"Will try to embed each sequence in the batch individually on the GPU. Full error: {e}"
                )
                for sequence in batch:
                    try:
                        yield from self._embed_batch_implementation([sequence], self._model.to(self._device))
                    except RuntimeError as e:
                        logger.warning(
                            f"RuntimeError for sequence with {len(sequence)} residues - falling back to CPU.. "
                            f"The error most likely means that you don't have enough "
                            f"GPU RAM to embed a protein this long on the GPU. Full error: {e}"
                        )
                        yield from self._embed_batch_implementation([sequence], self._get_fallback_model())
