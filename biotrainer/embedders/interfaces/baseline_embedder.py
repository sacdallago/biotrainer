import torch

from abc import ABC
from typing import Callable

from .embedder_interfaces import EmbedderInterface
from .preprocessing_strategies import preprocess_sequences_without_whitespaces

class BaselineEmbedder(EmbedderInterface, ABC):

    def _find_preprocessing_strategy(self) -> Callable:
        return preprocess_sequences_without_whitespaces


    @staticmethod
    def reduce_per_protein(embedding: torch.Tensor) -> torch.Tensor:
        """Returns the mean of all scale values across the sequence"""
        return embedding.mean(axis=0)
