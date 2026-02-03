import torch

from typing import List, Tuple, Optional, Dict, Iterable, Callable

from .preprocessing_strategies import (
    preprocess_sequences_with_whitespaces,
    preprocess_sequences_without_whitespaces,
)

from ...utilities import STANDARD_AAS


class BiotrainerTokenizerMixin:
    """
    Mixin to unify tokenization, preprocessing strategy selection, and amino-acid vocab mapping.

    Expects the host class to define:
      - self._tokenizer: a HuggingFace-like tokenizer with batch_encode_plus and get_vocab
      - self._model: a model with a .device attribute (used to place token tensors)
      - self.name: string identifier for model name (used for special-cases)
      - optional: self._preprocessing_strategy (callable). If not present, `_find_preprocessing_strategy` sets it.
    """

    def _tokenize(self, batch: List[str], preprocess: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if preprocess:
            batch = self._preprocess_sequences(batch)

        ids = self._tokenizer.batch_encode_plus(
            batch,
            add_special_tokens=True,
            is_split_into_words=False,
            padding="longest",
        )

        tokenized_sequences = torch.tensor(ids["input_ids"]).to(self._model.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self._model.device)
        return tokenized_sequences, attention_mask

    def _get_custom_indices_to_remove(self) -> List[int]:
        """
        Some embedders add specific tokens to the sequence, that cannot be identified via
        tokenizer.get_special_tokens_mask. If that is the case, they must be declared here manually.

        :return: List with indices that must be removed after the embedding was computed
        """
        return []  # Nothing by default

    def _find_preprocessing_strategy(self) -> Callable:
        # Default fallback implementation:
        # Try to automatically determine whether whitespaces are required
        dummy_sequence = [STANDARD_AAS]  # All 20 standard amino acids
        unknown_tokens = ["<unk>", "[UNK]", "UNK"]
        strategies = [
            preprocess_sequences_without_whitespaces,
            preprocess_sequences_with_whitespaces,
        ]

        for strategy in strategies:
            preprocessed = strategy(dummy_sequence, self.get_mask_token())
            tokenized, _ = self._tokenize(preprocessed)
            input_ids = tokenized[0].cpu().numpy()

            # Get the actual tokens
            tokens = self._tokenizer.convert_ids_to_tokens(input_ids)

            # Unknown tokens should not have been added
            if not any(unk in tokens for unk in unknown_tokens):
                # Check if the number of non-special tokens matches the original sequence length as a sanity check
                special_tokens_mask = self._tokenizer.get_special_tokens_mask(
                    input_ids, already_has_special_tokens=True
                )
                non_special_tokens = [token for token, mask in zip(tokens, special_tokens_mask) if mask == 0]

                if len(non_special_tokens) == len(dummy_sequence[0]):
                    return strategy

        # Default to no whitespace
        return preprocess_sequences_without_whitespaces

    def get_mask_token(self) -> Optional[str]:
        mask_token = "[MASK]"
        tokenizer_class_name = self._tokenizer.__class__.__name__.lower()
        if "t5" in tokenizer_class_name:
            mask_token = "<extra_id_0>"
        if "esm" in tokenizer_class_name:
            mask_token = "<mask>"
        return mask_token

    def get_mask_token_id(self) -> Optional[int]:
        try:
            mask_token_id = self._tokenizer.mask_token_id
            return mask_token_id
        except AttributeError:
            return self._tokenizer.added_tokens_encoder.get(self.get_mask_token(), None)

    def _preprocess_sequences(self, sequences: Iterable[str]) -> List[str]:
        return self._find_preprocessing_strategy()(sequences, self.get_mask_token())

    def aa_to_idx(self) -> Dict[str, int]:
        return {aa: self._tokenizer.get_vocab()[aa] for aa in STANDARD_AAS}
