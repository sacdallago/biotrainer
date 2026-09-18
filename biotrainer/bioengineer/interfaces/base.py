import torch
import numpy as np

from tqdm import tqdm
from typing import List, Optional
from abc import ABC, abstractmethod
from biotrainer_core.data_classes import VariantScore, Variant, SingleMutationScore, ZeroShotMethod

from ..bioengineer_utils import MAX_CONTEXT_LENGTH, convert_cat_jac_to_contact_map

from ...embedding.interfaces import BiotrainerTokenizerMixin


class BioEngineerModelWrapper(ABC, BiotrainerTokenizerMixin):

    def __init__(self, name: str, model, tokenizer, device: torch.device):
        self._name = name
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def max_context_length(self) -> int:
        """ Maximum number of tokens the model can process in one forward pass, including special tokens

        Only the categorical Jacobian guard honours this: the windowed marginal paths are hardcoded to
        WINDOW_SIZE-token windows, so a value below WINDOW_SIZE does not shrink them.
        """
        return MAX_CONTEXT_LENGTH

    @classmethod
    @abstractmethod
    def detect(cls, embedder_name: str, device: torch.device):
        raise NotImplementedError

    @abstractmethod
    def _model_forward_fn(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _model_batched_forward_fn(self, input_ids: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Batched forward pass separate from single-sequence forward pass. """
        raise NotImplementedError

    @abstractmethod
    def supported_methods(self) -> List[ZeroShotMethod]:
        """ Return a list of supported zero-shot methods """
        raise NotImplementedError

    @abstractmethod
    def _get_log_probabilities(self, sequence: str) -> torch.Tensor:
        """
        Get log probabilities for all positions without masking (WT-marginals).

        Returns:
            torch.Tensor: [seq_len, vocab_size]
        """
        raise NotImplementedError

    @abstractmethod
    def _get_masked_log_probabilities(self, sequence: str, batch_size: int = 32) -> torch.Tensor:
        """
        Get log probabilities for all positions using the masked-marginals strategy.
        Each position is masked independently and scored.

        Returns:
            torch.Tensor: [len(sequence), vocab_size] - residue positions only, no special tokens
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_pseudoperplexity(self, sequence: str) -> float:
        """
        Compute pseudoperplexity by extracting relevant log probs from masked logits.

        Returns:
            Sum of log P(aa_i | masked context) for all positions
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_perplexity(self, sequence: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def _compute_categorical_jacobian(self, sequence: str, batch_size: int = 32) -> torch.Tensor:
        """
        Compute categorical Jacobian from logits per sequence (L x 20 mutations).

        Returns:
            torch.Tensor (on CPU): [L, 20, L, 20]
        """

        raise NotImplementedError

    def _score_variants_from_marginal_probabilities(self,
                                                    wt_sequence: str,
                                                    log_probs: torch.Tensor,
                                                    mutations: List[str],
                                                    one_indexed: bool,
                                                    method_name: ZeroShotMethod) -> List[VariantScore]:
        aa_to_idx = self.aa_to_idx()

        variant_scores = []
        # Score mutations
        for mutation in tqdm(mutations, desc=f"Scoring mutations ({method_name.name})",
                             unit="variant", ncols=100, leave=True):
            variant = Variant.parse(mutation, wt_sequence=wt_sequence, one_indexed=one_indexed)
            mt_scores = []
            for mut in variant.mutations:
                wt_aa = mut.wt
                mt_aa = mut.mt

                mut_idx = mut.get_pos()
                assert wt_sequence[mut_idx] == wt_aa, (
                    f"Mismatch: position {mut_idx} in sequence is '{wt_sequence[mut_idx]}', not '{wt_aa}'"
                )

                wt_log_prob = log_probs[mut_idx, aa_to_idx[wt_aa]].item()
                mt_log_prob = log_probs[mut_idx, aa_to_idx[mt_aa]].item()

                mt_score = SingleMutationScore(mutation=mut, wt_log_prob=wt_log_prob, mt_log_prob=mt_log_prob)
                mt_scores.append(mt_score)

            variant_score = VariantScore.from_marginals(variant=variant, mutation_scores=mt_scores,
                                                        model_name=self._name, method_name=method_name)
            variant_scores.append(variant_score)

        return variant_scores

    def zero_shot_wt_marginals(self,
                               wt_sequence: str,
                               mutations: List[str],
                               one_indexed: Optional[bool] = True) -> List[VariantScore]:
        """
        Score mutations using the WT-marginals strategy (no masking).
        """
        log_probs = self._get_log_probabilities(wt_sequence)  # Raises NotImplementedError if not available
        return self._score_variants_from_marginal_probabilities(wt_sequence, log_probs, mutations, one_indexed,
                                                                ZeroShotMethod.WT_MARGINALS)

    def zero_shot_masked_marginals(self,
                                   wt_sequence: str,
                                   mutations: List[str],
                                   one_indexed: Optional[bool] = True,
                                   batch_size: int = 32) -> List[VariantScore]:
        """
        Score mutations using the masked-marginals strategy.
        Each position is independently masked and predicted.

        Args:
            wt_sequence: Wild-type protein sequence
            mutations: List of mutations to score
            one_indexed: Whether mutation positions are 1-indexed
            batch_size: Number of masked positions to score per forward pass

        Returns:
            List of VariantScore objects
        """
        log_probs = self._get_masked_log_probabilities(wt_sequence, batch_size)
        return self._score_variants_from_marginal_probabilities(wt_sequence, log_probs, mutations, one_indexed,
                                                                ZeroShotMethod.MASKED_MARGINALS)

    def zero_shot_pseudoperplexity(self,
                                   wt_sequence: str,
                                   mutations: List[str],
                                   one_indexed: Optional[bool] = True,
                                   subtract_wt: Optional[bool] = True) -> List[VariantScore]:
        """
        Score mutations using pseudoperplexity.

        Args:
            wt_sequence: Wild-type protein sequence
            mutations: List of mutations to score
            one_indexed: Whether mutation positions are 1-indexed
            subtract_wt: If True, return (MT_pppl - WT_pppl), else just MT_pppl

        Returns:
            List of VariantScore objects with pseudo-ppl scores
        """
        wt_pppl = None
        if subtract_wt:
            wt_pppl = self._compute_pseudoperplexity(wt_sequence)

        variant_scores = []
        for mutation in tqdm(mutations, desc="Scoring mutations (pseudo-ppl)",
                             unit="variant", ncols=100, leave=False):
            variant = Variant.parse(mutation, wt_sequence=wt_sequence, one_indexed=one_indexed)
            mutated_seq = variant.get_mutant_sequence()

            # Compute pseudo-ppl for mutant
            mt_pppl = self._compute_pseudoperplexity(mutated_seq)

            # Optionally subtract WT pseudo-ppl
            if subtract_wt:
                score = mt_pppl - wt_pppl
            else:
                score = mt_pppl

            variant_score = VariantScore.from_total_score(variant=variant, mutation_score=score,
                                                          model_name=self._name,
                                                          method_name=ZeroShotMethod.PSEUDOPERPLEXITY)
            variant_scores.append(variant_score)

        return variant_scores

    def zero_shot_perplexity(self, wt_sequence: str, mutations: List[str], one_indexed: bool = True,
                             subtract_wt: bool = True) -> List[VariantScore]:
        wt_ppl = self._compute_perplexity(wt_sequence) if subtract_wt else None
        results = []
        for mutation in tqdm(mutations, desc="Scoring mutations (ppl)", unit="variant", ncols=100, leave=False):
            variant = Variant.parse(mutation, wt_sequence=wt_sequence, one_indexed=one_indexed)
            mt_seq = variant.get_mutant_sequence()
            mt_ppl = self._compute_perplexity(mt_seq)
            score = mt_ppl - wt_ppl if subtract_wt else mt_ppl

            variant_score = VariantScore.from_total_score(variant=variant, mutation_score=score, model_name=self._name,
                                                          method_name=ZeroShotMethod.PERPLEXITY)
            results.append(variant_score)
        return results

    def zero_shot_contact_map_jacobian(self, sequence: str, batch_size: int = 32) -> np.ndarray:
        """
        Derive contact map from categorical Jacobian.

        Returns:
            np.ndarray: [L, L]
        """
        categorical_jacobian = self._compute_categorical_jacobian(sequence, batch_size)
        contact_map = convert_cat_jac_to_contact_map(categorical_jacobian.double().numpy())
        return contact_map
