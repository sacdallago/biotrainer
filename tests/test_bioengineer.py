import unittest

from biotrainer.bioengineer import BioEngineer, BioEngineerBaseline, ZeroShotMethod
from biotrainer.bioengineer.bioengineer_data_classes import Variant



class BioEngineerTests(unittest.TestCase):
    error_tolerance = 0.01

    def test_baselines(self):
        """ Test BioEngineer baselines on protein gym dataset """
        dataset_path = "test_input_files/pgym/B2L11_HUMAN_Dutta_2010_binding-Mcl-1.csv"
        # Check all baselines and methods
        for baseline in BioEngineerBaseline:
            for method in ZeroShotMethod:
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


