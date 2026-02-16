import torch

from tqdm import tqdm
from typing import List, Optional
from abc import ABC, abstractmethod

from .bioengineer_utils import compute_windowed_logits, get_optimal_window, MAX_CONTEXT_LENGTH
from .bioengineer_data_classes import VariantScore, Variant, SingleMutationScore, ZeroShotMethod

from ..embedders.interfaces import BiotrainerTokenizerMixin


class BioEngineerModelWrapper(ABC, BiotrainerTokenizerMixin):

    def __init__(self, name: str, model, tokenizer, device: torch.device):
        self._name = name
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    @classmethod
    @abstractmethod
    def detect(cls, embedder_name: str, device: torch.device):
        raise NotImplementedError

    @abstractmethod
    def _model_forward_fn(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def supported_methods(self) -> List[ZeroShotMethod]:
        """ Return a list of supported zero-shot methods """
        raise NotImplementedError

    @abstractmethod
    def _get_probabilities(self, sequence: str) -> torch.Tensor:
        """
        Get probabilities for all positions without masking (WT-marginals).

        Returns:
            torch.Tensor: [seq_len, vocab_size]
        """
        raise NotImplementedError

    @abstractmethod
    def _get_masked_probabilities(self, sequence: str) -> torch.Tensor:
        """
        Get probabilities for all positions using the masked-marginals strategy.
        Each position is masked independently and scored.

        Returns:
            torch.Tensor: [seq_len, vocab_size]
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
        log_probs = self._get_probabilities(wt_sequence)  # Raises NotImplementedError if not available
        return self._score_variants_from_marginal_probabilities(wt_sequence, log_probs, mutations, one_indexed,
                                                                ZeroShotMethod.WT_MARGINALS)

    def zero_shot_masked_marginals(self,
                                   wt_sequence: str,
                                   mutations: List[str],
                                   one_indexed: Optional[bool] = True) -> List[VariantScore]:
        """
        Score mutations using the masked-marginals strategy.
        Each position is independently masked and predicted.

        Args:
            wt_sequence: Wild-type protein sequence
            mutations: List of mutations to score
            one_indexed: Whether mutation positions are 1-indexed

        Returns:
            List of VariantScore objects
        """
        log_probs = self._get_masked_probabilities(wt_sequence)
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


class BertLikeEngineer(BioEngineerModelWrapper, ABC):
    """ Model wrapper for BERT-like models (e.g. ProtBert, ESM-2)

    Implementing classes should still overwrite the _find_preprocessing_strategy method for performance
    and consistency.
    """

    def supported_methods(self) -> List[ZeroShotMethod]:
        return [ZeroShotMethod.WT_MARGINALS, ZeroShotMethod.MASKED_MARGINALS, ZeroShotMethod.PSEUDOPERPLEXITY]

    def _model_forward_fn(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper to standardize model forward pass."""
        with torch.no_grad():
            output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return output.logits

    def _get_probabilities(self, sequence: str):
        tokenized_sequences, attention_mask = self._tokenize([sequence], preprocess=True)
        seq_len = tokenized_sequences.size(1)

        if seq_len > MAX_CONTEXT_LENGTH:
            # Returns log probabilities for entire sequence
            log_probs = compute_windowed_logits(
                sequence_tokens=tokenized_sequences,
                model_forward_fn=lambda ids, mask: self._model_forward_fn(ids, mask),
                attention_mask=attention_mask,
            )
        else:
            # Standard single-window scoring
            logits = self._model_forward_fn(input_ids=tokenized_sequences,
                                            attention_mask=attention_mask)
            logits = logits[0]  # [1, seq_len, vocab_size] -> [seq_len, vocab_size]
            logits = logits[1:-1]  # Remove EOS and BOS tokens
            # Use full vocabulary for probabilities (ProteinGym approach)
            log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs

    @staticmethod
    def _get_windowed_tokens(batch_tokens_masked: torch.Tensor, attention_mask: torch.Tensor,
                             masked_position: int, seq_len_with_special: int):
        start, end = get_optimal_window(
            masked_position=masked_position,
            seq_len_with_special=seq_len_with_special,
        )

        # Extract window
        windowed_tokens = batch_tokens_masked[:, start:end + 1]
        windowed_mask = attention_mask[:, start:end + 1] if attention_mask is not None else None
        return start, end, windowed_tokens, windowed_mask

    def _get_masked_probabilities(self, sequence: str) -> torch.Tensor:
        # Tokenize the sequence
        tokenized_sequences, attention_mask = self._tokenize([sequence], preprocess=True)

        # Get mask token ID
        mask_token_id = self.get_mask_token_id()

        all_token_probs = []

        # Iterate through all positions (excluding BOS and EOS)
        # tokenized_sequences[0, 0] = BOS
        # tokenized_sequences[0, 1:-1] = actual sequence
        # tokenized_sequences[0, -1] = EOS
        seq_len = tokenized_sequences.size(1)

        for i in tqdm(range(1, seq_len - 1), desc="Computing masked probabilities", unit="pos", ncols=100, leave=False):
            # Clone and mask position i
            batch_tokens_masked = tokenized_sequences.clone()
            batch_tokens_masked[0, i] = mask_token_id

            # Get optimal window
            start, end, windowed_tokens, windowed_mask = self._get_windowed_tokens(batch_tokens_masked, attention_mask,
                                                                                   masked_position=i,
                                                                                   seq_len_with_special=seq_len)
            logits = self._model_forward_fn(input_ids=windowed_tokens,
                                            attention_mask=windowed_mask)

            # Get log probabilities for the masked position
            token_logits = logits[0, i - start]  # [vocab_size]
            all_token_probs.append(token_logits.cpu())

        # Stack all position logits: [seq_len, vocab_size]
        logits = torch.stack(all_token_probs, dim=0)

        # Use full vocabulary for probabilities (ProteinGym approach)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs

    def _compute_pseudoperplexity(self, sequence: str) -> float:
        # Get masked probabilities for all positions
        log_probs = self._get_masked_probabilities(sequence)  # [seq_len, vocab_size]

        # Extract log probabilities for actual amino acids at each position
        aa_to_idx = self.aa_to_idx()

        position_log_probs = []
        for i, aa in enumerate(sequence):
            aa_idx = aa_to_idx[aa]
            position_log_probs.append(log_probs[i, aa_idx].item())

        # Return sum of log probabilities
        return sum(position_log_probs)

    def _compute_perplexity(self, sequence: str) -> float:
        raise NotImplementedError


class GPTLikeEngineer(BioEngineerModelWrapper, ABC):
    def supported_methods(self) -> List[ZeroShotMethod]:
        return [ZeroShotMethod.PERPLEXITY]

    def _get_probabilities(self, sequence: str):
        raise NotImplementedError("WT marginals are not defined for causal LMs")

    def _get_masked_probabilities(self, sequence: str):
        raise NotImplementedError("Masked marginals are not defined for causal LMs")

    def _compute_pseudoperplexity(self, sequence: str) -> float:
        raise NotImplementedError("Pseudo-ppl is for masked LMs; use perplexity for causal LMs")

    def _model_forward_fn(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            output = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            return output.loss  # mean cross-entropy per token (already shifted for causal LM)

    def _compute_perplexity(self, sequence: str) -> float:
        input_ids, attention_mask = self._tokenize([sequence], preprocess=True)
        loss = self._model_forward_fn(input_ids=input_ids, attention_mask=attention_mask)
        return torch.exp(loss).item()
