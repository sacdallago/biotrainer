import torch
import numpy as np

from enum import Enum
from typing import List, Dict, Optional

from .bioengineer_data_classes import ZeroShotMethod
from .bioengineer_interfaces import BioEngineerModelWrapper

from ..utilities import STANDARD_AAS


class BioEngineerBaseline(Enum):
    CONSTANT_BASELINE = "bioengineer_constant_baseline"
    RANDOM_BASELINE = "bioengineer_random_baseline"


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

    def _get_log_probabilities(self, sequence: str) -> torch.Tensor:
        return torch.full((len(sequence), 20), fill_value=self._log_prob, device=torch.device("cpu"))

    def _get_masked_log_probabilities(self, sequence: str) -> torch.Tensor:
        return torch.full((len(sequence), 20), fill_value=self._log_prob, device=torch.device("cpu"))

    def _compute_pseudoperplexity(self, sequence: str) -> float:
        """
        Pseudo-ppl for uniform distribution: log(1/20) per position.
        Sum over L positions: L * log(1/20) = L * log(0.05) ≈ -2.996 * L
        """
        return len(sequence) * self._log_prob

    def _compute_perplexity(self, sequence: str) -> float:
        return self._compute_pseudoperplexity(sequence)  # No difference here


class RandomEngineerBaseline(BioEngineerModelWrapper):
    """
    Baseline that assigns random probabilities to amino acids.

    Logits are sampled from N(0, 1), then converted to probabilities via softmax.
    Each position gets a random probability distribution over amino acids.

    Useful for testing that your evaluation pipeline can distinguish
    signal from noise.

    Args:
        seed: Random seed for reproducibility
    """

    _seed: int = 42

    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        if embedder_name in [BioEngineerBaseline.RANDOM_BASELINE.value, BioEngineerBaseline.RANDOM_BASELINE.name]:
            return cls(name=BioEngineerBaseline.RANDOM_BASELINE.value, model=None, tokenizer=None, device=device)
        return None

    def aa_to_idx(self) -> Dict[str, int]:
        return {aa: idx for idx, aa in enumerate(STANDARD_AAS)}

    def supported_methods(self) -> List[ZeroShotMethod]:
        return [ZeroShotMethod.WT_MARGINALS, ZeroShotMethod.MASKED_MARGINALS,
                ZeroShotMethod.PSEUDOPERPLEXITY, ZeroShotMethod.PERPLEXITY]

    def _model_forward_fn(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError  # Not necessary for baseline

    def _get_log_probabilities(self, sequence: str) -> torch.Tensor:
        """
        Return random probabilities sampled from N(0, 1).

        Note: We seed based on the sequence to ensure determinism
        (same sequence always gets same "random" logits).
        """
        # Seed based on sequence hash for reproducibility
        seq_seed = hash(sequence) % (2 ** 32)
        local_rng = np.random.RandomState(self._seed ^ seq_seed)

        # Sample logits from uniform distribution between 0 and 1
        logits = local_rng.uniform(0, 1, size=(len(sequence), 20)).astype(np.float32)
        logits = torch.from_numpy(logits)

        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs

    def _get_masked_log_probabilities(self, sequence: str) -> torch.Tensor:
        """
        For random baseline, masked-marginals same as wt-marginals.
        (No actual model to condition on context)
        """
        return self._get_log_probabilities(sequence)

    def _compute_pseudoperplexity(self, sequence: str) -> float:
        """
        Compute pseudo-ppl using the random logits.
        """
        log_probs = self._get_log_probabilities(sequence)

        aa_to_idx = self.aa_to_idx()
        position_log_probs = []

        for i, aa in enumerate(sequence):
            aa_idx = aa_to_idx[aa]
            position_log_probs.append(log_probs[i, aa_idx].item())

        return sum(position_log_probs)

    def _compute_perplexity(self, sequence: str) -> float:
        return self._compute_pseudoperplexity(sequence)  # No difference here
