import torch

from abc import ABC
from typing import List, Optional
from biotrainer_core.data_classes import ZeroShotMethod

from .base import BioEngineerModelWrapper


class GPTLikeEngineer(BioEngineerModelWrapper, ABC):
    def supported_methods(self) -> List[ZeroShotMethod]:
        return [ZeroShotMethod.PERPLEXITY]

    def _get_log_probabilities(self, sequence: str):
        raise NotImplementedError("WT marginals are not defined for causal LMs")

    def _get_masked_log_probabilities(self, sequence: str, batch_size: int = 32):
        raise NotImplementedError("Masked marginals are not defined for causal LMs")

    def _compute_pseudoperplexity(self, sequence: str) -> float:
        raise NotImplementedError("Pseudo-ppl is for masked LMs; use perplexity for causal LMs")

    def _model_batched_forward_fn(self, input_ids: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError("Batched forward pass is not defined for causal LMs")

    def _compute_categorical_jacobian(self, sequence: str, batch_size: int = 32) -> torch.Tensor:
        raise NotImplementedError("Categorical Jacobian is not defined for causal LMs")

    def _model_forward_fn(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            output = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            return output.loss  # mean cross-entropy per token (already shifted for causal LM)

    def _compute_perplexity(self, sequence: str) -> float:
        input_ids, attention_mask = self._tokenize([sequence], preprocess=True)
        loss = self._model_forward_fn(input_ids=input_ids, attention_mask=attention_mask)
        return torch.exp(loss).item()
