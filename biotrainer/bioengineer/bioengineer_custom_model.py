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


class CustomBioEngineerModelWrapper(BertLikeEngineer, GPTLikeEngineer):

    def __init__(self, custom_bioengineer: CustomBioEngineerModel, device: torch.device):
        super().__init__(name=custom_bioengineer.get_name(), model=None, tokenizer=None, device=device)
        self._custom_bioengineer = custom_bioengineer

    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        raise NotImplementedError  # Not necessary for custom model wrapper

    def _model_forward_fn(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._custom_bioengineer.run_model(input_ids, attention_mask)

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
