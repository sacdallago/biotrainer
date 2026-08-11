import unittest

from pathlib import Path

from biotrainer_core.data_classes import ZeroShotMethod, Variant
from biotrainer.bioengineer import BioEngineer, BioEngineerBaseline

import torch
import numpy as np

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from biotrainer_core.utils.constants import STANDARD_AAS
from biotrainer.bioengineer.bioengineer_interfaces import BertLikeEngineer
from biotrainer.bioengineer.bioengineer_custom_model import CustomBioEngineerModel, CustomBioEngineerModelWrapper
from biotrainer.shared import SequenceTooLongError
from biotrainer.shared.metrics import evaluate_contact_dataset
from biotrainer.bioengineer.bioengineer_utils import MAX_CONTEXT_LENGTH, WINDOW_SIZE

_BOS_TOKEN_ID = 0
_EOS_TOKEN_ID = 1
_REGISTER_TOKEN_ID = 2
_FIRST_AA_TOKEN_ID = 3
_MASK_TOKEN_ID = 25
_VOCAB_SIZE = 26


# One distinct logit row per token id. A lookup keeps the logits bit-identical whatever the input shape is,
# so the reference pass and the batched mutation passes can be compared for exact equality
_TOKEN_LOGITS = torch.sin(torch.arange(_VOCAB_SIZE).unsqueeze(-1) * (torch.arange(_VOCAB_SIZE) + 1.0))


class _LocalModel(torch.nn.Module):
    """ Logits at position p depend only on the token at p, so a mutation at residue i can only change
        the logits of residue i. Records every forward call so tests can assert on them. """

    def __init__(self):
        super().__init__()
        self.calls: List[torch.Tensor] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        self.calls.append(input_ids.clone())
        return SimpleNamespace(logits=_TOKEN_LOGITS[input_ids])


class _NoBosEngineer(BertLikeEngineer):
    """ Tokenizes as residues + EOS + one register token: no BOS, both special tokens at the end.

    This is the nanoPLM shape and A1's silent-failure case - the special tokens still count to two, so
    the Jacobian reshape succeeds while the mutation columns are off by one.
    """

    def __init__(self):
        super().__init__(name="no-bos", model=_LocalModel(), tokenizer=None, device=torch.device("cpu"))

    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        return None

    def _tokenize(self, batch: List[str], preprocess: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(batch) == 1, "Test engineer tokenizes one sequence at a time"
        aa_to_idx = self.aa_to_idx()
        token_ids = [aa_to_idx[aa] for aa in batch[0]] + [_EOS_TOKEN_ID, _REGISTER_TOKEN_ID]
        tokenized = torch.tensor([token_ids])
        return tokenized, torch.ones_like(tokenized)

    def _strip_special_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:-2]  # Drop EOS and the register token, there is no BOS

    def get_mask_token_id(self) -> Optional[int]:
        return _MASK_TOKEN_ID

    def aa_to_idx(self) -> Dict[str, int]:
        return {aa: _FIRST_AA_TOKEN_ID + idx for idx, aa in enumerate(STANDARD_AAS)}


class _WrongStripEngineer(_NoBosEngineer):
    """ strip_special_tokens forgets the register token, so it disagrees with its own tokenizer """

    def _strip_special_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:-1]


class _MisplacedStripEngineer(_NoBosEngineer):
    """ strip_special_tokens keeps the right number of positions, but the wrong ones: on the no-BOS layout
        the inherited [1:-1] drops the first residue and keeps the EOS token """

    _strip_special_tokens = BertLikeEngineer._strip_special_tokens


class _ReorderingStripEngineer(_NoBosEngineer):
    """ strip_special_tokens keeps exactly the residue positions, but in reverse order """

    def _strip_special_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:-2].flip(0)


class _SplitEngineer(_NoBosEngineer):
    """ Tokenizes as BOS + first half + register + second half + EOS: the residue token positions are
        not contiguous, and strip_special_tokens is an index select instead of a slice """

    def _tokenize(self, batch: List[str], preprocess: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(batch) == 1, "Test engineer tokenizes one sequence at a time"
        aa_to_idx = self.aa_to_idx()
        residue_ids = [aa_to_idx[aa] for aa in batch[0]]
        half = len(residue_ids) // 2
        token_ids = ([_BOS_TOKEN_ID] + residue_ids[:half] + [_REGISTER_TOKEN_ID] + residue_ids[half:] +
                     [_EOS_TOKEN_ID])
        tokenized = torch.tensor([token_ids])
        return tokenized, torch.ones_like(tokenized)

    def _strip_special_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        half = (tensor.shape[0] - 3) // 2  # n_tokens = BOS + L residues + register + EOS
        keep = list(range(1, half + 1)) + list(range(half + 2, tensor.shape[0] - 1))
        return tensor[torch.tensor(keep)]


class _ShortContextEngineer(_NoBosEngineer):
    """ A model with a tiny context, so the length guard can be tested without a 1000-residue sequence """

    def max_context_length(self) -> int:
        return 8


class _StandardEngineer(_NoBosEngineer):
    """ Tokenizes as BOS + residues + EOS: the layout every real in-repo engineer uses """

    # Deliberately the inherited default ([1:-1]), not _NoBosEngineer's trailing-two variant
    _strip_special_tokens = BertLikeEngineer._strip_special_tokens

    def _tokenize(self, batch: List[str], preprocess: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(batch) == 1, "Test engineer tokenizes one sequence at a time"
        aa_to_idx = self.aa_to_idx()
        token_ids = [_BOS_TOKEN_ID] + [aa_to_idx[aa] for aa in batch[0]] + [_EOS_TOKEN_ID]
        tokenized = torch.tensor([token_ids])
        return tokenized, torch.ones_like(tokenized)


class _MinimalCustomModel(CustomBioEngineerModel):
    """ A custom model that implements nothing but its name - only the context length is exercised here """

    # Overriding the remaining abstract hooks with None keeps the class instantiable without stub bodies
    supported_methods = run_model = strip_special_tokens = tokenize = get_mask_token_id = \
        preprocess_sequences = aa_to_idx = None

    def get_name(self) -> str:
        return "minimal-custom"


class _LongContextCustomModel(_MinimalCustomModel):
    """ A custom model whose context is not the ESM-style 1024 tokens """

    def max_context_length(self) -> int:
        return 2048


class BioEngineerTests(unittest.TestCase):
    error_tolerance = 0.01
    # No two consecutive residues are equal, so an off-by-one in the Jacobian is always visible
    jacobian_sequence = "MAGSMALMKQ"

    def test_baselines(self):
        """ Test BioEngineer baselines on protein gym dataset """
        dataset_path = "test_input_files/pgym/B2L11_HUMAN_Dutta_2010_binding-Mcl-1.csv"
        # Check all baselines and methods
        for baseline in BioEngineerBaseline:
            for method in ZeroShotMethod:
                if method == ZeroShotMethod.JACOBIAN_CONTACT:
                    continue
                bio_engineer = BioEngineer.from_baseline(baseline=baseline)
                if method not in bio_engineer.model_wrapper.supported_methods():
                    continue
                self.assertIsNotNone(bio_engineer.model_wrapper, f"Model wrapper for baseline {baseline} is None!")
                scores, ranking = bio_engineer.rank_pgym_dataset(dataset_file_path=dataset_path,
                                                                 method=method)
                self.assertTrue(len(scores) > 0)
                self.assertTrue(-1 <= ranking.scc.mean <= 1)
                self.assertTrue(0 <= ranking.ndcg.mean <= 1)

        # Check that baseline creation from name works
        bio_engineer = BioEngineer.from_name(name=BioEngineerBaseline.CONSTANT_BASELINE.name)
        scores, ranking = bio_engineer.rank_pgym_dataset(dataset_file_path=dataset_path,
                                                         method=ZeroShotMethod.WT_MARGINALS)
        self.assertTrue(len(scores) > 0)
        self.assertTrue(-1 <= ranking.scc.mean <= 1)
        self.assertTrue(0 <= ranking.ndcg.mean <= 1)

    def test_mutation_parsing(self):
        wt_sequence = "MAGSMALM"
        mutations = ["A2S", "G3M", "M1A", "M1S", "M1S:M5A"]
        for mutation in mutations:
            variant = Variant.parse(variant_string=mutation, wt_sequence=wt_sequence, one_indexed=True)
            self.assertEqual(variant.wt_sequence, wt_sequence)
            self.assertEqual(len(variant.mutations), 1 if ":" not in mutation else 2)

            mut_seq = variant.get_mutant_sequence(wt_sequence=wt_sequence)
            self.assertTrue(len(mut_seq) == len(wt_sequence))

    def test_masked_marginals_only_masks_residue_positions(self):
        """ One forward per residue, each masking that residue - special tokens are never masked """
        engineer = _NoBosEngineer()
        sequence = "MAGSMALMKQ"

        log_probs = engineer._get_masked_log_probabilities(sequence)

        self.assertEqual(tuple(log_probs.shape), (len(sequence), _VOCAB_SIZE))
        calls = engineer._model.calls
        self.assertEqual(len(calls), len(sequence),
                         "Special tokens were masked and scored, wasting a forward pass each!")
        for residue_index, call in enumerate(calls):
            masked_positions = (call[0] == _MASK_TOKEN_ID).nonzero().flatten().tolist()
            self.assertEqual(masked_positions, [residue_index],
                             f"Forward pass {residue_index} masked {masked_positions} instead")

    def test_masked_marginals_masks_the_residues_token_position(self):
        """ BOS + residues + EOS: the mask follows the residue's token position, not its sequence index.

        _NoBosEngineer cannot catch this - its residues start at token 0, so residue index and token
        position coincide and a loop over range(len(sequence)) would look correct.
        """
        engineer = _StandardEngineer()
        sequence = "MAGSMALMKQ"

        engineer._get_masked_log_probabilities(sequence)

        calls = engineer._model.calls
        self.assertEqual(len(calls), len(sequence))
        for residue_index, call in enumerate(calls):
            masked_positions = (call[0] == _MASK_TOKEN_ID).nonzero().flatten().tolist()
            self.assertEqual(masked_positions, [residue_index + 1],  # +1 for the BOS token
                             f"Forward pass {residue_index} masked {masked_positions} instead")

    def test_masked_marginals_reads_the_masked_position_inside_the_window(self):
        """ Past WINDOW_SIZE tokens the forward pass only sees a window, so the row has to be read at i - start.

        _LocalModel's logits at position p depend only on the token at p, so every returned row must be the
        row of the mask token: a read at the wrong offset lands on a residue token and yields a different row.
        """
        engineer = _NoBosEngineer()
        sequence = self.jacobian_sequence * 110  # 1100 residues + 2 special tokens, past WINDOW_SIZE

        log_probs = engineer._get_masked_log_probabilities(sequence)

        self.assertEqual(tuple(log_probs.shape), (len(sequence), _VOCAB_SIZE))
        calls = engineer._model.calls
        self.assertEqual(len(calls), len(sequence))
        self.assertEqual(calls[0].shape[1], WINDOW_SIZE, "The windowed path was not exercised!")

        masked_row = _TOKEN_LOGITS[_MASK_TOKEN_ID].log_softmax(dim=-1)
        misread = (log_probs != masked_row).any(dim=-1).nonzero().flatten().tolist()
        self.assertEqual(misread, [], "These rows were not read at the masked position of their window!")

    def test_masked_marginals_masks_non_contiguous_residue_positions(self):
        """ BOS + half + register + half + EOS: the mask has to skip the register token in the middle """
        engineer = _SplitEngineer()
        sequence = self.jacobian_sequence
        half = len(sequence) // 2

        log_probs = engineer._get_masked_log_probabilities(sequence)

        self.assertEqual(tuple(log_probs.shape), (len(sequence), _VOCAB_SIZE))
        calls = engineer._model.calls
        self.assertEqual(len(calls), len(sequence))
        for residue_index, call in enumerate(calls):
            masked_positions = (call[0] == _MASK_TOKEN_ID).nonzero().flatten().tolist()
            # +1 for BOS, and one more once the register token in the middle has been passed
            expected = residue_index + 1 + (1 if residue_index >= half else 0)
            self.assertEqual(masked_positions, [expected],
                             f"Forward pass {residue_index} masked {masked_positions} instead")

    def test_wt_marginals_do_not_window_below_the_window_size(self):
        """ A model whose context is below WINDOW_SIZE must not enter the windowed path: windowing is
            hardcoded to WINDOW_SIZE tokens and crashes on anything shorter """
        engineer = _ShortContextEngineer()  # 8-token context, far below WINDOW_SIZE
        sequence = self.jacobian_sequence

        log_probs = engineer._get_log_probabilities(sequence)

        self.assertEqual(tuple(log_probs.shape), (len(sequence), _VOCAB_SIZE))

    def _assert_jacobian_is_aligned(self, engineer: BertLikeEngineer, sequence: str):
        """ Row i of the Jacobian must describe residue i, for any tokenizer layout.

        The discriminating assertion is the identity mutation: replacing residue i with itself reproduces the
        reference logits exactly, so that row of the Jacobian is all zeros - for any model, not just this fake
        one. If the mutation was written to the wrong token, the row for the wild-type amino acid is not zero.

        A locality check alone would NOT catch this bug: with residues + EOS + register, the write offset
        (n+1) and the read offset ([1:-1]) shift by the same amount and cancel on the diagonal.
        """
        jac = engineer._compute_categorical_jacobian(sequence, batch_size=8)

        seq_len = len(sequence)
        self.assertEqual(tuple(jac.shape), (seq_len, 20, seq_len, 20))
        for i, residue in enumerate(sequence):
            wt_index = STANDARD_AAS.index(residue)
            self.assertEqual(jac[i, wt_index].abs().max().item(), 0.0,
                             f"Jacobian row {i} is not aligned to residue {i} ({residue}): mutating it to "
                             f"itself changed the logits!")
            for aa_index in range(20):
                if aa_index == wt_index:
                    continue
                self.assertGreater(jac[i, aa_index, i].abs().max().item(), 0.0,
                                   f"Mutating residue {i} to {STANDARD_AAS[aa_index]} left its own logits "
                                   f"untouched!")

        # This fake model is local, so a mutation may never move another residue's logits
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    continue
                self.assertEqual(jac[i, :, j, :].abs().max().item(), 0.0,
                                 f"Mutating residue {i} changed the logits of residue {j}!")

    def test_categorical_jacobian_is_aligned_to_the_right_residue(self):
        """ residues + EOS + register: both special tokens trailing, so a hardcoded [1:-1] misaligns """
        self._assert_jacobian_is_aligned(_NoBosEngineer(), self.jacobian_sequence)

    def test_categorical_jacobian_is_aligned_for_the_standard_layout(self):
        """ BOS + residues + EOS through the default strip_special_tokens: the layout real engineers use """
        self._assert_jacobian_is_aligned(_StandardEngineer(), self.jacobian_sequence)

    def test_categorical_jacobian_is_aligned_for_non_contiguous_residues(self):
        """ BOS + half + register + half + EOS: the residue positions are not a contiguous slice """
        self._assert_jacobian_is_aligned(_SplitEngineer(), self.jacobian_sequence)

    def test_categorical_jacobian_rejects_misplaced_strip_special_tokens(self):
        """ The right number of positions taken from the wrong places must fail as loudly as the wrong count,
            otherwise the contact map is silently misaligned """
        sequence = self.jacobian_sequence

        for engineer in [_MisplacedStripEngineer(), _ReorderingStripEngineer()]:
            with self.subTest(engineer=type(engineer).__name__):
                with self.assertRaises(ValueError) as raised:
                    engineer._compute_categorical_jacobian(sequence, batch_size=8)

                message = str(raised.exception)
                self.assertIn("residue 0", message)  # Only the first mismatch is actionable
                self.assertIn(str(engineer.aa_to_idx()[sequence[0]]), message)  # Token id expected there

    def test_categorical_jacobian_rejects_disagreeing_strip_special_tokens(self):
        """ A strip_special_tokens that disagrees with the tokenizer must fail loudly, not silently """
        engineer = _WrongStripEngineer()
        sequence = self.jacobian_sequence

        with self.assertRaises(ValueError) as raised:
            engineer._compute_categorical_jacobian(sequence, batch_size=8)

        # The counts are what makes the message actionable: it keeps one position too many
        message = str(raised.exception)
        self.assertIn(str(len(sequence) + 1), message)  # Positions this strip wrongly kept
        self.assertIn(str(len(sequence)), message)  # Residues actually in the sequence

    def test_categorical_jacobian_rejects_too_long_sequences(self):
        """ The Jacobian has no windowed variant, so an over-long sequence must be rejected, not truncated """
        engineer = _ShortContextEngineer()  # 10 residues + 2 special tokens > 8

        with self.assertRaises(SequenceTooLongError):
            engineer._compute_categorical_jacobian("MAGSMALMKQ", batch_size=8)

    def test_categorical_jacobian_allows_the_exact_context_length(self):
        """ The guard is exclusive, so a sequence that fills the context exactly must still be computed """
        engineer = _ShortContextEngineer()  # 6 residues + 2 special tokens == 8
        jac = engineer._compute_categorical_jacobian("MAGSMA", batch_size=8)
        self.assertEqual(tuple(jac.shape), (6, 20, 6, 20))

    def test_default_max_context_length(self):
        self.assertEqual(_NoBosEngineer().max_context_length(), MAX_CONTEXT_LENGTH)

    def test_random_baseline_rejects_too_long_sequences(self):
        """ The baseline samples an [L, 20, L, 20] array, so it has to reject before allocating: a MemoryError
            is not a SequenceTooLongError, so the contact evaluator could not skip the protein """
        baseline = BioEngineer.from_baseline(baseline=BioEngineerBaseline.RANDOM_BASELINE).model_wrapper
        sequence = "A" * (baseline.max_context_length() + 1)

        with self.assertRaises(SequenceTooLongError):
            baseline._compute_categorical_jacobian(sequence)


class CustomModelContextLengthTests(unittest.TestCase):
    """ The Jacobian guard asks the wrapper, so a custom model's declared context has to reach it """

    def test_context_length_defaults_and_overrides_reach_the_wrapper(self):
        self.assertEqual(_MinimalCustomModel().max_context_length(), MAX_CONTEXT_LENGTH)
        wrapper = CustomBioEngineerModelWrapper(custom_bioengineer=_LongContextCustomModel(),
                                                device=torch.device("cpu"))
        self.assertEqual(wrapper.max_context_length(), 2048)


class ContactDatasetEvaluationTests(unittest.TestCase):
    """ evaluate_contact_dataset resumes from cache, so one over-long protein must not kill the run forever """

    @staticmethod
    def _symmetric_target(rng, seq_len: int = 40) -> np.ndarray:
        target = (rng.random((seq_len, seq_len)) > 0.9).astype(float)
        return np.maximum(target, target.T)

    def test_too_long_proteins_are_skipped(self):
        rng = np.random.default_rng(0)
        target = self._symmetric_target(rng)

        def predict(item):
            if item == "too_long":
                raise SequenceTooLongError("2000 tokens exceed the 1024 token context")
            return rng.random(target.shape)

        results = list(evaluate_contact_dataset(dataset_name="test",
                                                items=["too_long", "fine"],
                                                predict_func=predict,
                                                get_ground_truth_func=lambda item: target,
                                                get_seq_id_func=lambda item: item))

        self.assertEqual([result.protein_name for result in results], ["fine"])

    def test_other_errors_still_propagate(self):
        """ The skip must be narrow - a real bug in predict_func may not be swallowed.

        ValueError is the case that matters: SequenceTooLongError subclasses it and _residue_token_positions
        raises a plain one on this very call path, so a catch widened to ValueError would silently skip a
        tokenizer/strip_special_tokens disagreement instead of failing loudly.
        """
        rng = np.random.default_rng(0)
        target = self._symmetric_target(rng)

        for error in [RuntimeError("boom"), ValueError("strip_special_tokens disagrees with the tokenizer")]:
            with self.subTest(error=type(error).__name__):
                def predict(item):
                    raise error

                with self.assertRaises(type(error)):
                    list(evaluate_contact_dataset(dataset_name="test",
                                                  items=["a"],
                                                  predict_func=predict,
                                                  get_ground_truth_func=lambda item: target,
                                                  get_seq_id_func=lambda item: item))
