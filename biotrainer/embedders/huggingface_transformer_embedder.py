# Inspired by bio_embeddings embed module (https://github.com/sacdallago/bio_embeddings/tree/develop/bio_embeddings/embed)

import torch
import numpy as np

from typing import List, Generator, Any, Union, Tuple
from numpy import ndarray

from .embedder_interfaces import EmbedderWithFallback
from .preprocessing_strategies import preprocess_sequences_with_whitespaces, preprocess_sequences_without_whitespaces, \
    preprocess_sequences_for_prostt5

from ..utilities import get_logger

logger = get_logger(__name__)


class HuggingfaceTransformerEmbedder(EmbedderWithFallback):

    def __init__(self, name: str, model, tokenizer, use_half_precision: bool, device: Union[str, torch.device]):
        self.name = name
        self._model = model
        self._tokenizer = tokenizer
        self._use_half_precision = use_half_precision
        self._device = device
        self._preprocessing_strategy = self._find_preprocessing_strategy()
        self._custom_indices_to_remove = self._get_custom_indices_to_remove()
        self._set_model_precision()

    def _find_preprocessing_strategy(self):
        # Handle special ProstT5 case
        if self.name == "Rostlab/ProstT5":
            strategy = preprocess_sequences_for_prostt5
            logger.info(f"Chosen sequence pre-processing strategy: {strategy.__name__}")
            return strategy

        # Other models
        dummy_sequence = ["ACDEFGHIKLMNPQRSTVWY"]  # All 20 standard amino acids
        unknown_tokens = ["<unk>", "[UNK]", "UNK"]
        strategies = [preprocess_sequences_without_whitespaces, preprocess_sequences_with_whitespaces]

        for strategy in strategies:
            preprocessed = strategy(dummy_sequence)
            tokenized, _ = self._tokenize(preprocessed)
            input_ids = tokenized[0].cpu().numpy()

            # Get the actual tokens
            tokens = self._tokenizer.convert_ids_to_tokens(input_ids)

            # Unknown tokens should not have been added
            if not any(unk in tokens for unk in unknown_tokens):
                # Check if the number of non-special tokens matches the original sequence length as a sanity check
                special_tokens_mask = self._tokenizer.get_special_tokens_mask(input_ids,
                                                                              already_has_special_tokens=True)
                non_special_tokens = [token for token, mask in zip(tokens, special_tokens_mask) if mask == 0]

                if len(non_special_tokens) == len(dummy_sequence[0]):
                    logger.info(f"Chosen sequence pre-processing strategy: {strategy.__name__}")
                    return strategy
                else:
                    logger.debug(
                        f"Token count mismatch. Expected {len(dummy_sequence[0])}, got {len(non_special_tokens)}")

        logger.warning("Could not determine correct sequence pre-processing strategy, defaulting to no whitespace.")
        return preprocess_sequences_without_whitespaces

    def _get_custom_indices_to_remove(self) -> List[int]:
        """
        Some embedders add specific tokens to the sequence, that cannot be identified via
        tokenizer.get_special_tokens_mask. If that is the case, they must be declared here manually.

        :return: List with indices that must be removed after the embedding was computed
        """
        if self.name == "Rostlab/ProstT5":
            return [0]
        return []

    def _set_model_precision(self):
        if self._use_half_precision and self._device == "cpu":
            # This is caught earlier, but we check it here again for safety
            raise NotImplementedError("Cannot use half_precision mode together with cpu!")
        if self._use_half_precision:
            try:
                self._model = self._model.half()
            except AttributeError:
                raise NotImplementedError(f"Given model {self.name} does not support half_precision mode!")

    def _get_fallback_model(self):
        """ Returns the CPU model """
        if self._use_half_precision:
            raise NotImplementedError(
                "You sequence was too long for the GPU, "
                "but we can't fall back to the CPU with half_precision_model=True "
                "(https://github.com/huggingface/transformers/issues/11546)"
            )
        return self._model.to("cpu")

    def _embed_single(self, sequence: str) -> ndarray:
        [embedding] = self._embed_batch([sequence])
        return embedding

    def _tokenize(self, batch: List[str]) -> Tuple[torch.tensor, torch.tensor]:
        ids = self._tokenizer.batch_encode_plus(batch, add_special_tokens=True,
                                                is_split_into_words=False,
                                                padding="longest")

        tokenized_sequences = torch.tensor(ids["input_ids"]).to(self._model.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self._model.device)
        return tokenized_sequences, attention_mask

    def _remove_special_tokens(self, embedding: ndarray, input_id: ndarray) -> ndarray:
        """
        Remove special tokens from the embedding.

        :param embedding: The per-residue embedding for a single sequence
        :param input_id: The input ids for the sequence
        :return: The embedding with special token indices removed
        """
        special_tokens_mask = self._tokenizer.get_special_tokens_mask(input_id, already_has_special_tokens=True)
        indices_to_remove = [index for index, mask in enumerate(special_tokens_mask) if mask != 0]
        indices_to_remove += self._custom_indices_to_remove
        indices_to_remove = list(set(indices_to_remove))
        return np.delete(embedding, indices_to_remove, axis=0)

    def _embed_batch_implementation(self, batch: List[str], model: Any) -> Generator[ndarray, None, None]:
        tokenized_sequences, attention_mask = self._tokenize(batch)

        with torch.no_grad():
            embeddings = model(
                input_ids=tokenized_sequences,
                attention_mask=attention_mask,
            )

        embeddings = embeddings[0].cpu().numpy()
        for seq_num in range(len(embeddings)):
            input_id = tokenized_sequences[seq_num].cpu().numpy()
            embedding = self._remove_special_tokens(embeddings[seq_num], input_id)
            yield embedding
