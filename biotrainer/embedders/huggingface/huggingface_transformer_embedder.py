# Inspired by bio_embeddings embed module (https://github.com/sacdallago/bio_embeddings/tree/develop/bio_embeddings/embed)
import torch

from peft import PeftModel
from typing import List, Generator, Any, Union, Optional

from ..interfaces import EmbedderWithFallback


from ...utilities import get_logger

logger = get_logger(__name__)


class HuggingfaceTransformerEmbedder(EmbedderWithFallback):
    """ Generic Huggingface Transformer Embedder"""

    def __init__(self, name: str, model, tokenizer, use_half_precision: bool, device: Union[str, torch.device]):
        self.name = name
        self._model = model
        self._tokenizer = tokenizer
        self._use_half_precision = use_half_precision
        self._device = device
        self._custom_indices_to_remove = self._get_custom_indices_to_remove()
        self._mask_token_id: Optional[int] = self.get_mask_token_id()
        self._set_model_precision()

    def _set_model_precision(self):
        if self._use_half_precision and self._device == "cpu":
            # This is caught earlier, but we check it here again for safety
            raise NotImplementedError("Cannot use half_precision mode together with cpu!")
        if self._use_half_precision:
            try:
                self._model = self._model.half()
            except AttributeError:
                raise NotImplementedError(f"Given model {self.name} does not support half_precision mode!")

    def _get_gradient_context(self):
        if isinstance(self._model, PeftModel) and self._model.training:
            return torch.enable_grad()  # Finetuning
        return torch.no_grad()  # Usual embeddings inference

    def _get_fallback_model(self):
        """ Returns the CPU model """
        if self._use_half_precision:
            raise NotImplementedError(
                "You sequence was too long for the GPU, "
                "but we can't fall back to the CPU with half_precision_model=True "
                "(https://github.com/huggingface/transformers/issues/11546)"
            )
        return self._model.to("cpu")

    def _embed_single(self, sequence: str) -> torch.Tensor:
        [embedding] = self._embed_batch([sequence])
        return embedding

    def _remove_special_tokens(self, embedding: torch.Tensor, input_id: torch.Tensor) -> torch.Tensor:
        """
        Remove special tokens from the embedding.

        :param embedding: The per-residue embedding for a single sequence
        :param input_id: The input ids for the sequence
        :return: The embedding with special token indices removed
        """
        special_tokens_mask = self._tokenizer.get_special_tokens_mask(input_id, already_has_special_tokens=True)
        # Replace all special tokens but the mask token for MLM
        indices_to_remove = [index for index, mask in enumerate(special_tokens_mask)
                             if mask != 0 and input_id[index] != self._mask_token_id]
        indices_to_remove += self._custom_indices_to_remove
        indices_to_remove = list(set(indices_to_remove))

        # Create a boolean mask for indices to keep
        keep_mask = torch.ones(embedding.size(0), dtype=torch.bool, device=embedding.device)
        keep_mask[indices_to_remove] = False

        return embedding[keep_mask]

    def _embed_batch_implementation(self, batch: List[str], model: Any) -> Generator[
        torch.Tensor, None, None]:
        tokenized_sequences, attention_mask = self._tokenize(batch)

        with self._get_gradient_context():
            embeddings = model(
                input_ids=tokenized_sequences,
                attention_mask=attention_mask,
            )

        embeddings = embeddings.last_hidden_state
        # Process all sequences before yielding (keeps them on GPU longer)
        processed_embeddings = []
        for seq_num in range(len(embeddings)):
            input_id = tokenized_sequences[seq_num]
            embedding = self._remove_special_tokens(embeddings[seq_num], input_id)
            processed_embeddings.append(embedding)

        # Yield all at once
        yield from processed_embeddings
