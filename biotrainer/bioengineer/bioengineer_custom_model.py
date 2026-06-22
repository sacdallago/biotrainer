import torch

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Iterable, Tuple

from .bioengineer_data_classes import ZeroShotMethod
from .bioengineer_interfaces import BertLikeEngineer, GPTLikeEngineer


class CustomBioEngineerModel(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """ Return the name of the custom bioengineer model """
        raise NotImplementedError

    @abstractmethod
    def supported_methods(self) -> List[ZeroShotMethod]:
        """ Return a list of supported zero-shot methods for this custom bioengineer model """
        raise NotImplementedError

    @abstractmethod
    def run_model(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run the model on a batch of input IDs

        Should return either output.logits (for BERT-like models) or output.loss (for GPT-like models)
        """
        raise NotImplementedError

    @abstractmethod
    def strip_special_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        """ Remove special tokens from output (if necessary) """
        return tensor

    @abstractmethod
    def tokenize(self, batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Tokenize a batch of sequences"""
        raise NotImplementedError

    @abstractmethod
    def get_mask_token_id(self) -> Optional[int]:
        """ Get the ID of the mask token"""
        raise NotImplementedError

    @abstractmethod
    def preprocess_sequences(self, sequences: Iterable[str]) -> List[str]:
        """ Preprocess a batch of sequences """
        raise NotImplementedError

    @abstractmethod
    def aa_to_idx(self) -> Dict[str, int]:
        """ Return a dictionary mapping amino acids to their indices """
        raise NotImplementedError

    def run_model_batched(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run the model on a batch of sequences.

        The default implementation loops over each sequence in the batch and calls run_model, 
        then stacks the results. This is for backward compatibility with older versions of bioengineer.
        Override this method for a more efficient native batched forward pass.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Optional mask of shape [batch_size, seq_len]

        Returns:
            torch.Tensor of shape [batch_size, seq_len, vocab_size]
        """
        batch_size = input_ids.size(0)
        results = []
        for i in range(batch_size):
            single_ids = input_ids[i].unsqueeze(0)  # [1, seq_len]
            single_mask = attention_mask[i].unsqueeze(0) if attention_mask is not None else None
            logits = self.run_model(single_ids, single_mask)  # [seq_len, vocab_size]
            results.append(logits)
        return torch.stack(results, dim=0)  # [batch_size, seq_len, vocab_size]


class CustomBioEngineerModelWrapper(BertLikeEngineer, GPTLikeEngineer):

    def __init__(self, custom_bioengineer: CustomBioEngineerModel, device: torch.device):
        super().__init__(name=custom_bioengineer.get_name(), model=None, tokenizer=None, device=device)
        self._custom_bioengineer = custom_bioengineer

    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        raise NotImplementedError  # Not necessary for custom model wrapper

    def _model_forward_fn(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._custom_bioengineer.run_model(input_ids, attention_mask)

    def _model_batched_forward_fn(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._custom_bioengineer.run_model_batched(input_ids, attention_mask)
        # NOTE: for backward compatibility, run_model expected to return single sequence results; for batched results, run_model_batched to be used or re-implemented!
        # TODO: cleanest fix for future is to have only run_model & _model_forward_fn, both handling batches by default, and a wrapper on top to extract logit/loss for index [0] as required!

    def _strip_special_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._custom_bioengineer.strip_special_tokens(tensor)

    def supported_methods(self) -> List[ZeroShotMethod]:
        return self._custom_bioengineer.supported_methods()

    def _tokenize(self, batch: List[str], preprocess: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if preprocess:
            batch = self._custom_bioengineer.preprocess_sequences(batch)
        return self._custom_bioengineer.tokenize(batch)

    def get_mask_token_id(self) -> Optional[int]:
        return self._custom_bioengineer.get_mask_token_id()

    def _preprocess_sequences(self, sequences: Iterable[str]) -> List[str]:
        return self._custom_bioengineer.preprocess_sequences(sequences)

    def aa_to_idx(self) -> Dict[str, int]:
        return self._custom_bioengineer.aa_to_idx()
