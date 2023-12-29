# Inspired by bio_embeddings embed module (https://github.com/sacdallago/bio_embeddings/tree/develop/bio_embeddings/embed)

import torch
import regex as re

from typing import List, Generator, Any, Union
from numpy import ndarray

from .embedder_interfaces import EmbedderWithFallback


class HuggingfaceTransformerEmbedder(EmbedderWithFallback):

    def __init__(self, name: str, model, tokenizer, use_half_precision: bool, device: Union[str, torch.device]):
        self.name = name
        self._model = model
        self._tokenizer = tokenizer
        self._use_half_precision = use_half_precision
        self._device = device

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

    @staticmethod
    def _preprocess_sequences(sequences: List[str]) -> List[str]:
        # Remove rare amino acids
        sequences_cleaned = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
        # Transformers need spaces between the amino acids
        sequences_with_spaces = [" ".join(list(sequence)) for sequence in sequences_cleaned]
        return sequences_with_spaces

    def _embed_batch_implementation(self, batch: List[str], model: Any) -> Generator[ndarray, None, None]:
        ids = self._tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest")

        tokenized_sequences = torch.tensor(ids["input_ids"]).to(self._device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self._device)
        model.to(self._device)

        with torch.no_grad():
            embeddings = model(
                input_ids=tokenized_sequences,
                attention_mask=attention_mask,
            )

        embeddings = embeddings[0].cpu().numpy()
        for seq_num in range(len(embeddings)):
            # slice off last position (special token)
            embedding = embeddings[seq_num][:-1]
            yield embedding
