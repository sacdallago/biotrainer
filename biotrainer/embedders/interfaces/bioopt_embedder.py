import torch

from typing import Optional
from abc import ABC, abstractmethod


class BioOptEmbedder(ABC):
    """ Biocentral-Optimized Embedder Interface """

    @classmethod
    def detect(cls, embedder_name: str, use_half_precision: bool, dtype: torch.dtype, device: torch.device) -> Optional:
        """ Detect if this embedder should be used and construct it """
        raise NotImplementedError

    @abstractmethod
    def embedding_dim(self) -> int:
        """ Return the embedding dimension of this embedder """
        raise NotImplementedError
