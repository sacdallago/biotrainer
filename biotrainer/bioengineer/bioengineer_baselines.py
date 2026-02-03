import torch
import numpy as np

from enum import Enum
from typing import List, Dict

from .bioengineer_data_classes import ZeroShotMethod
from .bioengineer_interfaces import BioEngineerModelWrapper

from ..utilities import STANDARD_AAS


class BioengineerBaseline(Enum):
    CONSTANT = "CONSTANT"
    RANDOM = "RANDOM"


class ConstantEngineerBaseline(BioEngineerModelWrapper):

    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        if embedder_name.upper() == BioengineerBaseline.CONSTANT.value:
            return cls(name=embedder_name, model=None, tokenizer=None)
        return None

    def aa_to_idx(self) -> Dict[str, int]:
        return {aa: idx for idx, aa in enumerate(STANDARD_AAS)}

    def supported_methods(self) -> List[ZeroShotMethod]:
        return [ZeroShotMethod.WT_MARGINALS, ZeroShotMethod.MASKED_MARGINALS, ZeroShotMethod.PSEUDOPERPLEXITY]

    def _get_logits(self, sequence: str) -> torch.Tensor:
        return torch.full((len(sequence), 20), fill_value=0.2, device=torch.device("cpu"))

    def _get_masked_logits(self, sequence: str) -> torch.Tensor:
        return torch.full((len(sequence), 20), fill_value=0.2, device=torch.device("cpu"))

    def _compute_pseudoperplexity(self, sequence: str) -> float:
        """
        Pseudo-ppl for uniform distribution: log(1/20) per position.
        Sum over L positions: L * log(1/20) = L * log(0.05) ≈ -2.996 * L
        """
        return len(sequence) * np.log(1.0 / 20)


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
        if embedder_name.upper() == BioengineerBaseline.RANDOM.value:
            return cls(name=embedder_name, model=None, tokenizer=None)
        return None

    def aa_to_idx(self) -> Dict[str, int]:
        return {aa: idx for idx, aa in enumerate(STANDARD_AAS)}

    def supported_methods(self) -> List[ZeroShotMethod]:
        return [ZeroShotMethod.WT_MARGINALS, ZeroShotMethod.MASKED_MARGINALS, ZeroShotMethod.PSEUDOPERPLEXITY]

    def _get_logits(self, sequence: str) -> torch.Tensor:
        """
        Return random logits sampled from N(0, 1).

        Note: We seed based on the sequence to ensure determinism
        (same sequence always gets same "random" logits).
        """
        # Seed based on sequence hash for reproducibility
        seq_seed = hash(sequence) % (2 ** 32)
        local_rng = np.random.RandomState(self._seed ^ seq_seed)

        # Sample logits from standard normal distribution
        logits = local_rng.randn(len(sequence), 20).astype(np.float32)

        return torch.from_numpy(logits)

    def _get_masked_logits(self, sequence: str) -> torch.Tensor:
        """
        For random baseline, masked-marginals same as wt-marginals.
        (No actual model to condition on context)
        """
        return self._get_logits(sequence)

    def _compute_pseudoperplexity(self, sequence: str) -> float:
        """
        Compute pseudo-ppl using the random logits.
        """
        logits = self._get_logits(sequence)
        log_probs = torch.log_softmax(logits, dim=-1)

        aa_to_idx = self.aa_to_idx()
        position_log_probs = []

        for i, aa in enumerate(sequence):
            aa_idx = aa_to_idx[aa]
            position_log_probs.append(log_probs[i, aa_idx].item())

        return sum(position_log_probs)