import torch
import numpy as np

from tqdm import tqdm
from typing import List, Optional
from abc import ABC, abstractmethod
from biotrainer_core.data_classes import VariantScore, Variant, SingleMutationScore, ZeroShotMethod

from .bioengineer_utils import compute_windowed_logits, get_optimal_window, MAX_CONTEXT_LENGTH, WINDOW_SIZE, \
    prepare_cat_jac_mutations, convert_cat_jac_to_contact_map

from ..embedding.interfaces import BiotrainerTokenizerMixin
from ..shared.sequence_exception import SequenceTooLongError


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
    def _get_masked_log_probabilities(self, sequence: str) -> torch.Tensor:
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
        log_probs = self._get_masked_log_probabilities(wt_sequence)
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
        contact_map = convert_cat_jac_to_contact_map(categorical_jacobian.numpy().astype(np.float64))
        return contact_map


class BertLikeEngineer(BioEngineerModelWrapper, ABC):
    """ Model wrapper for BERT-like models (e.g. ProtBert, ESM-2)

    Implementing classes should still overwrite the _find_preprocessing_strategy method for performance
    and consistency.
    """

    def supported_methods(self) -> List[ZeroShotMethod]:
        return [ZeroShotMethod.WT_MARGINALS, ZeroShotMethod.MASKED_MARGINALS, ZeroShotMethod.PSEUDOPERPLEXITY,
                ZeroShotMethod.JACOBIAN_CONTACT]

    def _model_forward_fn(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper to standardize model forward pass."""
        with torch.no_grad():
            output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = output.logits
            logits = logits[0]  # [1, seq_len, vocab_size] -> [seq_len, vocab_size]
        return logits

    def _model_batched_forward_fn(self, input_ids: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper to standardize model batched forward pass."""
        with torch.no_grad():
            output = self._model(input_ids=input_ids, attention_mask=attention_mask)
            return output.logits

    def _strip_special_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[1:-1]  # Remove BOS and EOS tokens

    def _residue_token_positions(self, input_ids: torch.Tensor, sequence: str) -> torch.Tensor:
        """ Token positions holding the real residues, derived from whatever _strip_special_tokens removes.

        Keeps every special-token offset in one place: a model that does not tokenize as BOS + residues + EOS
        only has to implement strip_special_tokens correctly, which it already must for the marginal methods.
        Checked both by count and by value, so a strip that keeps the right number of positions but takes
        them from the wrong places - the silent misalignment - fails here instead of in the contact map.
        """
        n_tokens = input_ids.shape[1]
        # 2-D probe, so a strip written against the [seq_len, vocab] logits (tensor[1:-1, :]) works here too.
        # On input_ids' device, not self._device: the positions exist to index input_ids, and a custom model's
        # tokenize() is free to return CPU tensors while the wrapper holds an accelerator
        probe = torch.arange(n_tokens, device=input_ids.device).unsqueeze(-1)  # [n_tokens, 1]
        positions = self._strip_special_tokens(probe).flatten()
        if positions.numel() != len(sequence):
            raise ValueError(
                f"Tokenizer of {self._name} produced {n_tokens} tokens, of which {positions.numel()} are "
                f"residue positions, but the sequence has {len(sequence)} residues. strip_special_tokens "
                f"and the tokenizer disagree about which tokens are special."
            )
        aa_to_idx = self.aa_to_idx()
        found_ids = input_ids[0, positions].tolist()
        for index, residue in enumerate(sequence):
            # Only standard amino acids have a reliable expectation: preprocessing maps UZOB to X, and
            # anything outside STANDARD_AAS may well tokenize to the unknown token
            if residue in aa_to_idx and found_ids[index] != aa_to_idx[residue]:
                raise ValueError(
                    f"Tokenizer of {self._name} has token {found_ids[index]} at derived residue position "
                    f"{positions[index].item()}, but residue {index} of the sequence is '{residue}', which "
                    f"tokenizes to {aa_to_idx[residue]}. strip_special_tokens and the tokenizer disagree "
                    f"about which tokens are special."
                )
        return positions

    def _get_log_probabilities(self, sequence: str):
        tokenized_sequences, attention_mask = self._tokenize([sequence], preprocess=True)
        seq_len = tokenized_sequences.size(1)

        if seq_len > WINDOW_SIZE:  # windowing is hardcoded to WINDOW_SIZE, so the decision has to match it
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
            logits = self._strip_special_tokens(logits)
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
        # get_optimal_window returns a half-open interval [start, end)
        windowed_tokens = batch_tokens_masked[:, start:end]
        windowed_mask = attention_mask[:, start:end] if attention_mask is not None else None
        return start, end, windowed_tokens, windowed_mask

    def _get_masked_log_probabilities(self, sequence: str) -> torch.Tensor:
        # Tokenize the sequence
        tokenized_sequences, attention_mask = self._tokenize([sequence], preprocess=True)

        # Get mask token ID
        mask_token_id = self.get_mask_token_id()

        all_token_probs = []

        # Iterate the token positions that hold real residues, so no forward pass is spent on a special
        # token whose row would be stripped from the result anyway
        seq_len = tokenized_sequences.size(1)
        residue_positions = self._residue_token_positions(tokenized_sequences, sequence)

        for i in tqdm(residue_positions.tolist(), desc="Computing masked probabilities", unit="pos", ncols=100,
                      leave=False):
            # Clone and mask position i
            batch_tokens_masked = tokenized_sequences.clone()
            batch_tokens_masked[0, i] = mask_token_id

            # Get optimal window
            start, end, windowed_tokens, windowed_mask = self._get_windowed_tokens(batch_tokens_masked, attention_mask,
                                                                                   masked_position=i,
                                                                                   seq_len_with_special=seq_len)
            logits = self._model_forward_fn(input_ids=windowed_tokens,
                                            attention_mask=windowed_mask)  # [n_window_tokens, vocab_size]

            # Get log probabilities for the masked position
            token_position = i - start
            token_logits = logits[token_position]  # [vocab_size]
            all_token_probs.append(token_logits.cpu())

        # Stack all position logits: [len(sequence), vocab_size]
        logits = torch.stack(all_token_probs, dim=0)

        # Use full vocabulary for probabilities (ProteinGym approach)
        # No stripping needed: only residue positions were scored
        return torch.log_softmax(logits, dim=-1)

    def _compute_pseudoperplexity(self, sequence: str) -> float:
        # Get masked probabilities for all positions
        log_probs = self._get_masked_log_probabilities(sequence)  # [len(sequence), vocab_size]

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

    def _compute_categorical_jacobian(self, sequence: str, batch_size: int = 32) -> torch.Tensor:
        # Tokenize the sequence
        input_ids, attention_mask = self._tokenize([sequence], preprocess=True)
        # Get the IDs of the amino acids in order of STANDARD_AAS, on input_ids' device: they are written into
        # a tile of input_ids, and a custom model's tokenize() may well return CPU tensors
        aa_token_ids = torch.tensor(list(self.aa_to_idx().values()), device=input_ids.device)
        n_tokens = input_ids.shape[1]
        if n_tokens > self.max_context_length():
            raise SequenceTooLongError(
                f"Sequence of {len(sequence)} residues tokenizes to {n_tokens} tokens, which exceeds the "
                f"{self.max_context_length()} token context of {self._name}. The categorical Jacobian has no "
                f"windowed variant - contacts spanning two windows would be missing - so it is not computed."
            )
        # Which token positions hold the actual residues - no BOS/EOS arrangement is assumed
        residue_positions = self._residue_token_positions(input_ids, sequence)
        # For each position in the sequence, prepare the input with all mutations
        mutated_inputs, mutated_mask = prepare_cat_jac_mutations(input_ids, attention_mask, aa_token_ids,
                                                                residue_positions)

        # Get the model's logits without mutations
        ref_logits = self._model_forward_fn(input_ids, attention_mask)
        # Remove the special tokens and keep only the logits for amino acids
        ref_logits = ref_logits[residue_positions][:, aa_token_ids].cpu()
        # Compute the logits for all mutations
        mutated_logits = []
        for batch_ids, batch_mask in zip(torch.split(mutated_inputs, batch_size),
                                         torch.split(mutated_mask, batch_size)):
            mut_logits = self._model_batched_forward_fn(batch_ids, batch_mask)
            mutated_logits.append(mut_logits[:, residue_positions][:, :, aa_token_ids].cpu())
        L = residue_positions.numel()
        # [L*20, L, 20] -> [L, 20, L, 20] in order of aa_token_ids/STANDARD_AAS
        mutated_logits = torch.cat(mutated_logits, dim=0).reshape(L, 20, L, 20)

        # Compute the jacobian
        jac = mutated_logits - ref_logits
        return jac


class GPTLikeEngineer(BioEngineerModelWrapper, ABC):
    def supported_methods(self) -> List[ZeroShotMethod]:
        return [ZeroShotMethod.PERPLEXITY]

    def _get_log_probabilities(self, sequence: str):
        raise NotImplementedError("WT marginals are not defined for causal LMs")

    def _get_masked_log_probabilities(self, sequence: str):
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
