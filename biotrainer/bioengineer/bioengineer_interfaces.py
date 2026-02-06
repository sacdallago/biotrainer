import torch

from tqdm import tqdm
from typing import List, Optional
from abc import ABC, abstractmethod

from .bioengineer_data_classes import VariantScore, Variant, SingleMutationScore, ZeroShotMethod

from ..embedders.interfaces import BiotrainerTokenizerMixin


class BioEngineerModelWrapper(ABC, BiotrainerTokenizerMixin):

    def __init__(self, name, model, tokenizer):
        self._name = name
        self._model = model
        self._tokenizer = tokenizer

    @classmethod
    @abstractmethod
    def detect(cls, embedder_name: str, device: torch.device):
        raise NotImplementedError

    @abstractmethod
    def supported_methods(self) -> List[ZeroShotMethod]:
        """ Return a list of supported zero-shot methods """
        raise NotImplementedError

    @abstractmethod
    def _get_logits(self, sequence: str) -> torch.Tensor:
        """
        Get logits for all positions without masking (WT-marginals).

        Returns:
            torch.Tensor: [seq_len, vocab_size]
        """
        raise NotImplementedError

    @abstractmethod
    def _get_masked_logits(self, sequence: str) -> torch.Tensor:
        """
        Get logits for all positions using the masked-marginals strategy.
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

    def _score_variants_from_marginal_logits(self,
                                             wt_sequence: str,
                                             logits: torch.Tensor,
                                             mutations: List[str],
                                             one_indexed: bool,
                                             method_name: ZeroShotMethod) -> List[VariantScore]:
        # Use full vocabulary (ProteinGym approach)
        log_probs = torch.log_softmax(logits, dim=-1)
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
        logits = self._get_logits(wt_sequence)  # Raises NotImplementedError if not available
        return self._score_variants_from_marginal_logits(wt_sequence, logits, mutations, one_indexed,
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
        logits = self._get_masked_logits(wt_sequence)
        return self._score_variants_from_marginal_logits(wt_sequence, logits, mutations, one_indexed,
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

    def _get_logits(self, sequence: str):
        tokenized_sequences, attention_mask = self._tokenize([sequence], preprocess=True)

        with torch.no_grad():
            output = self._model(
                input_ids=tokenized_sequences,
                attention_mask=attention_mask,
            )

        logits = output.logits[0]  # [1, seq_len, vocab_size] -> [seq_len, vocab_size]
        logits = logits[1:-1]  # Remove EOS and BOS tokens
        return logits

    def _get_masked_logits(self, sequence: str) -> torch.Tensor:
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

        for i in tqdm(range(1, seq_len - 1), desc="Computing masked logits", unit="pos", ncols=100, leave=False):
            # Clone and mask position i
            batch_tokens_masked = tokenized_sequences.clone()
            batch_tokens_masked[0, i] = mask_token_id

            with torch.no_grad():
                output = self._model(
                    input_ids=batch_tokens_masked,
                    attention_mask=attention_mask,
                )

                # Get log probabilities for the masked position
                token_logits = output.logits[0, i]  # [vocab_size]
                all_token_probs.append(token_logits.cpu())

        # Stack all position logits: [seq_len, vocab_size]
        logits = torch.stack(all_token_probs, dim=0)

        return logits

    def _compute_pseudoperplexity(self, sequence: str) -> float:
        # Get masked logits for all positions
        logits = self._get_masked_logits(sequence)  # [seq_len, vocab_size]

        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)  # [seq_len, vocab_size]

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

    def _get_logits(self, sequence: str):
        raise NotImplementedError("WT marginals are not defined for causal LMs")

    def _get_masked_logits(self, sequence: str):
        raise NotImplementedError("Masked marginals are not defined for causal LMs")

    def _compute_pseudoperplexity(self, sequence: str) -> float:
        raise NotImplementedError("Pseudo-ppl is for masked LMs; use perplexity for causal LMs")

    def _compute_perplexity(self, sequence: str) -> float:
        input_ids, attention_mask = self._tokenize([sequence], preprocess=True)
        with torch.no_grad():
            out = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss  # mean cross-entropy per token (already shifted for causal LM)
        return torch.exp(loss).item()
