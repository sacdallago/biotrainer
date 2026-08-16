import torch
import unittest

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from biotrainer_core.data_classes import ZeroShotMethod, Variant
from biotrainer.bioengineer import BioEngineer, BioEngineerBaseline
from biotrainer.bioengineer.bioengineer_interfaces import BertLikeEngineer

_VOCAB_SIZE = 25
_MASK_TOKEN_ID = 24


class _DeterministicModel(torch.nn.Module):
    """ Model whose logits depend only on the token at a position, so that batching cannot change the result """

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        vocab = torch.arange(_VOCAB_SIZE, dtype=torch.float32).view(1, 1, -1)
        logits = torch.sin(input_ids.unsqueeze(-1).float() * (vocab + 1.0))
        return type("Output", (), {"logits": logits})


class _DeterministicEngineer(BertLikeEngineer):
    """ BertLikeEngineer over a deterministic model, to test masked-marginals without downloading weights """

    def __init__(self):
        super().__init__(name="deterministic", model=_DeterministicModel(), tokenizer=None,
                         device=torch.device("cpu"))

    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        return None

    def _tokenize(self, batch: List[str], preprocess: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = [[0] + [1 + (ord(aa) % 20) for aa in batch[0]] + [1]]
        tokenized_sequences = torch.tensor(token_ids)
        return tokenized_sequences, torch.ones_like(tokenized_sequences)

    def get_mask_token_id(self) -> Optional[int]:
        return _MASK_TOKEN_ID

    def aa_to_idx(self) -> Dict[str, int]:
        return {}


class BioEngineerTests(unittest.TestCase):
    error_tolerance = 0.01

    def test_masked_marginals_batching(self):
        """ Batched masked-marginals scoring must be equivalent to scoring one masked position at a time """
        engineer = _DeterministicEngineer()
        sequence = "MAGSMALMKQWERTYIPDFNHCV" * 3

        unbatched = engineer._get_masked_log_probabilities(sequence, batch_size=1)
        batched = engineer._get_masked_log_probabilities(sequence, batch_size=32)

        self.assertEqual(batched.shape, (len(sequence), _VOCAB_SIZE))
        self.assertTrue(torch.equal(batched, unbatched),
                        "Batched masked-marginals differ from the unbatched reference!")

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
