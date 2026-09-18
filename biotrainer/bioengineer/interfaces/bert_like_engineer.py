import torch

from abc import ABC
from tqdm import tqdm
from typing import List, Optional
from biotrainer_core.data_classes import ZeroShotMethod

from .base import BioEngineerModelWrapper

from ..bioengineer_utils import compute_windowed_logits, get_optimal_window, WINDOW_SIZE, \
    prepare_cat_jac_mutations

from ...shared.sequence_exception import SequenceTooLongError


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

    def _get_masked_log_probabilities(self, sequence: str, batch_size: int = 32) -> torch.Tensor:
        # Tokenize the sequence
        tokenized_sequences, attention_mask = self._tokenize([sequence], preprocess=True)

        # Get mask token ID
        mask_token_id = self.get_mask_token_id()

        # Iterate the token positions that hold real residues, so no forward pass is spent on a special
        # token whose row would be stripped from the result anyway
        seq_len = tokenized_sequences.size(1)
        residue_positions = self._residue_token_positions(tokenized_sequences, sequence)

        # Collect the scoring window of every masked position. get_optimal_window returns windows of uniform width
        # within a sequence, so all masked variants stack into a single [len(sequence), window_size] tensor.
        windowed_tokens, windowed_masks, masked_positions = [], [], []
        for i in tqdm(residue_positions.tolist(), desc="Computing masked probabilities", unit="pos", ncols=100,
                      leave=False):
            # Clone and mask position i
            batch_tokens_masked = tokenized_sequences.clone()
            batch_tokens_masked[0, i] = mask_token_id

            # Get optimal window
            start, end, window_tokens, window_mask = self._get_windowed_tokens(batch_tokens_masked, attention_mask,
                                                                               masked_position=i,
                                                                               seq_len_with_special=seq_len)
            windowed_tokens.append(window_tokens[0])
            windowed_masks.append(window_mask[0] if window_mask is not None else None)
            masked_positions.append(i - start)

        windowed_tokens = torch.stack(windowed_tokens, dim=0)  # [len(sequence), window_size]
        windowed_masks = None if windowed_masks[0] is None else torch.stack(windowed_masks, dim=0)
        masked_positions = torch.tensor(masked_positions)

        # The masked positions are scored independently of each other, so their forward passes can be batched
        token_splits = torch.split(windowed_tokens, batch_size)
        mask_splits = torch.split(windowed_masks, batch_size) if windowed_masks is not None else [None] * len(
            token_splits)
        position_splits = torch.split(masked_positions, batch_size)

        all_token_probs = []
        for batch_tokens, batch_mask, batch_positions in tqdm(zip(token_splits, mask_splits, position_splits),
                                                              desc="Computing masked probabilities", unit="batch",
                                                              total=len(token_splits), ncols=100, leave=False):
            logits = self._model_batched_forward_fn(input_ids=batch_tokens,
                                                    attention_mask=batch_mask)  # [batch, window_size, vocab_size]

            # Get the logits of every window at its own masked position
            batch_positions = batch_positions.to(logits.device)
            batch_indices = torch.arange(batch_positions.size(0), device=logits.device)
            all_token_probs.append(logits[batch_indices, batch_positions].cpu())  # [batch, vocab_size]

        # Concatenate all position logits: [len(sequence), vocab_size]
        logits = torch.cat(all_token_probs, dim=0)

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
                f"windowed variant and contacts spanning two windows would be missing, so it is not computed."
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
