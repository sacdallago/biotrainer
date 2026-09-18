import torch
import numpy as np

from typing import List, Dict, Optional
from biotrainer_core.data_classes import ZeroShotMethod
from biotrainer_core.utils.constants import STANDARD_AAS

from .base import BioEngineerBaseline

from ..interfaces import BioEngineerModelWrapper


class ConstantEngineerBaseline(BioEngineerModelWrapper):
    _log_prob = np.log(0.05)

    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        if embedder_name in [BioEngineerBaseline.CONSTANT_BASELINE.value, BioEngineerBaseline.CONSTANT_BASELINE.name]:
            return cls(name=BioEngineerBaseline.CONSTANT_BASELINE.value, model=None, tokenizer=None, device=device)
        return None

    def aa_to_idx(self) -> Dict[str, int]:
        return {aa: idx for idx, aa in enumerate(STANDARD_AAS)}

    def supported_methods(self) -> List[ZeroShotMethod]:
        return [ZeroShotMethod.WT_MARGINALS, ZeroShotMethod.MASKED_MARGINALS, ZeroShotMethod.PSEUDOPERPLEXITY,
                ZeroShotMethod.PERPLEXITY]

    def _model_forward_fn(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError  # Not necessary for baseline

    def _model_batched_forward_fn(self, input_ids: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError  # Not necessary for baseline

    def _get_log_probabilities(self, sequence: str) -> torch.Tensor:
        return torch.full((len(sequence), 20), fill_value=self._log_prob, device=torch.device("cpu"))

    def _get_masked_log_probabilities(self, sequence: str, batch_size: int = 32) -> torch.Tensor:
        return torch.full((len(sequence), 20), fill_value=self._log_prob, device=torch.device("cpu"))

    def _compute_pseudoperplexity(self, sequence: str) -> float:
        """
        Pseudo-ppl for uniform distribution: log(1/20) per position.
        Sum over L positions: L * log(1/20) = L * log(0.05) ≈ -2.996 * L
        """
        return len(sequence) * self._log_prob

    def _compute_perplexity(self, sequence: str) -> float:
        return self._compute_pseudoperplexity(sequence)  # No difference here

    def _compute_categorical_jacobian(self, sequence: str, batch_size: int = 32) -> torch.Tensor:
        raise NotImplementedError("Categorical Jacobian is not defined for constant baseline")