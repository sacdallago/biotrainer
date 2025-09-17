# Inspired by bio_embeddings embed module (https://github.com/sacdallago/bio_embeddings/tree/develop/bio_embeddings/embed)
import torch
import numpy as np

from peft import PeftModel
from typing import List, Generator, Any, Union, Tuple, Dict

from ..interfaces import (EmbedderWithFallback, preprocess_sequences_with_whitespaces,
                         preprocess_sequences_without_whitespaces, preprocess_sequences_for_prostt5)


from ...utilities import get_logger

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
        self._mask_token_id: int = self._get_mask_token_id()
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
            preprocessed = strategy(dummy_sequence, self.get_mask_token())
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

    def get_mask_token(self) -> str:
        mask_token = "[MASK]"
        tokenizer_class_name = self._tokenizer.__class__.__name__.lower()
        if "t5" in tokenizer_class_name:
            mask_token = "<extra_id_0>"
        if "esm" in tokenizer_class_name:
            mask_token = "<mask>"
        return mask_token

    def _get_mask_token_id(self) -> int:
        return self._tokenizer.added_tokens_encoder[self.get_mask_token()]

    def _get_aa_token_id(self, amino_acid: str) -> int:
        """Get token ID for a single amino acid."""
        # Handle the preprocessing - some models might need whitespace
        if self._preprocessing_strategy == preprocess_sequences_with_whitespaces:
            test_seq = f"A {amino_acid} A"  # Embed in context
            tokenized = self._tokenizer.encode(test_seq, add_special_tokens=False)
            # The amino acid should be the middle token
            if len(tokenized) >= 3:
                return tokenized[1]

        # Fallback: direct tokenization
        tokenized = self._tokenizer.encode(amino_acid, add_special_tokens=False)
        if len(tokenized) > 0:
            return tokenized[0]

        raise ValueError(f"Could not tokenize amino acid: {amino_acid}")

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

    def _embed_single(self, sequence: str) -> torch.tensor:
        [embedding] = self._embed_batch([sequence])
        return embedding

    def _tokenize(self, batch: List[str]) -> Tuple[torch.tensor, torch.tensor]:
        ids = self._tokenizer.batch_encode_plus(batch, add_special_tokens=True,
                                                is_split_into_words=False,
                                                padding="longest")

        tokenized_sequences = torch.tensor(ids["input_ids"]).to(self._model.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self._model.device)
        return tokenized_sequences, attention_mask

    def _remove_special_tokens(self, embedding: torch.tensor, input_id: torch.tensor) -> torch.tensor:
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

    def _embed_batch_implementation(self, batch: List[str], model: Any) -> Generator[torch.tensor, None, None]:
        tokenized_sequences, attention_mask = self._tokenize(batch)

        with self._get_gradient_context():
            embeddings = model(
                input_ids=tokenized_sequences,
                attention_mask=attention_mask,
            )

        embeddings = embeddings[0]
        for seq_num in range(len(embeddings)):
            input_id = tokenized_sequences[seq_num]
            embedding = self._remove_special_tokens(embeddings[seq_num], input_id)
            yield embedding

    def get_position_probabilities(self, sequence: str, position: int) -> Dict[str, float]:
        """
        [BETA]
        Get probability distribution over all amino acids at a specific position. This only works if the model
        provides logits in the output, like BERT models do (e.g. Rostlab/prot_bert)

        Args:
            sequence: The protein sequence
            position: 0-based position to get probabilities for

        Returns:
            List of probabilities for each amino acid (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y)
        """
        # Create masked sequence
        sequence_list = list(sequence)
        sequence_list[position] = self.get_mask_token()
        masked_sequence = ''.join(sequence_list)

        # Preprocess and tokenize
        preprocessed = self._preprocessing_strategy([masked_sequence], self.get_mask_token())
        tokenized_sequences, attention_mask = self._tokenize(preprocessed)

        # Find the mask token position in tokenized sequence
        mask_position_in_tokens = self._find_mask_position_in_tokens(
            tokenized_sequences[0]
        )

        if mask_position_in_tokens is None:
            raise ValueError(f"Could not find mask token")

        # Get logits
        with torch.no_grad():
            outputs = self._model(
                input_ids=tokenized_sequences,
                attention_mask=attention_mask,
            )

        # Get logits for the masked position
        logits = outputs.logits[0, mask_position_in_tokens]  # Shape: [vocab_size]
        probs = torch.softmax(logits, dim=-1)

        # Get probabilities for all 20 amino acids in standard order
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                       'Y']
        aa_probabilities = {}

        for aa in amino_acids:
            token_id = self._get_aa_token_id(aa)
            aa_probabilities[aa] = probs[token_id].item()

        return aa_probabilities

    def _find_mask_position_in_tokens(self, tokenized_sequence: torch.tensor) -> Union[int, None]:
        """
        Find the position of the mask token in the tokenized sequence.
        This accounts for special tokens and preprocessing strategy.
        """
        input_ids = tokenized_sequence.cpu().numpy()

        # Find mask token
        mask_positions = np.where(input_ids == self._mask_token_id)[0]
        if len(mask_positions) == 0:
            return None

        # For most cases, there should be exactly one mask token
        if len(mask_positions) == 1:
            return int(mask_positions[0])

        # If multiple masks (shouldn't happen in our case), return the first one
        logger.warning(f"Found multiple mask tokens, using the first one..")
        return int(mask_positions[0])
