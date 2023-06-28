import unittest

from biotrainer.inference import Inferencer
from bio_embeddings.embed import OneHotEncodingEmbedder


class ConfigurationVerificationTests(unittest.TestCase):
    _test_sequences = ["SEQVENCEPRTEI",
                       "SEQWENCE",
                       "PRTEIN"]
    _test_targets_r2c = ["CDEVVCCCDDDDD",
                         "DDDDDDDD",
                         "CDEVVV"]
    _test_classes_r2c = ["C", "D", "E", "F", "V"]
    _test_targets_s2v = [1, -1.212, 0.0]

    def setUp(self) -> None:
        self.inferencer_r2c, _ = Inferencer.create_from_out_file("test_input_files/test_models/r2c/out.yml")
        self.inferencer_s2v, _ = Inferencer.create_from_out_file("test_input_files/test_models/s2v/out.yml")
        self.per_residue_embeddings, self.per_sequence_embeddings = self._embed()

    def _embed(self):
        embedder = OneHotEncodingEmbedder()
        per_residue_embeddings = list(embedder.embed_many(self._test_sequences))
        per_sequence_embeddings = [[embedder.reduce_per_protein(embedding)] for embedding in
                                   per_residue_embeddings]
        per_residue_embeddings_dict = {f"Seq{idx}": embed for idx, embed in enumerate(per_residue_embeddings)}
        per_sequence_embeddings_dict = {f"Seq{idx}": embed for idx, embed in enumerate(per_sequence_embeddings)}
        return per_residue_embeddings_dict, per_sequence_embeddings_dict

    def test_from_embeddings(self):
        r2c_dict = self.inferencer_r2c.from_embeddings(self.per_residue_embeddings)
        s2v_dict = self.inferencer_s2v.from_embeddings(self.per_sequence_embeddings)

        self.assertTrue("metrics" in r2c_dict.keys() and "metrics" in s2v_dict.keys(), "Missing metrics key!")
        self.assertTrue("mapped_predictions" in r2c_dict.keys() and "mapped_predictions" in s2v_dict.keys(),
                        "Missing predictions key!")

        self.assertTrue(all([pred_class in self._test_classes_r2c for pred_class in
                             "".join(r2c_dict["mapped_predictions"].values())]),
                        "Inferencer predicted a non existing class!")
        self.assertTrue(all([type(value) is float for value in s2v_dict["mapped_predictions"].values()]),
                        "Type of all sequence to value predictions is not float!")
        print(r2c_dict)
        print(s2v_dict)

    def test_from_embeddings_with_targets(self):
        r2c_dict = self.inferencer_r2c.from_embeddings(self.per_residue_embeddings, self._test_targets_r2c)
        s2v_dict = self.inferencer_s2v.from_embeddings(self.per_sequence_embeddings, self._test_targets_s2v)

        self.assertAlmostEqual(r2c_dict["metrics"]["loss"], 1.6270031929016113, msg="Loss not as expected for r2c!")
        self.assertAlmostEqual(s2v_dict["metrics"]["loss"], 0.8427120447158813, msg="Loss not as expected for s2v!")

    def test_from_embeddings_with_bootstrapping(self):
        """
        Checks that metrics calculated with and without bootstrapping are within about the same range
        """
        error_tolerance_factor = 2
        r2c_dict_bootstrapping = self.inferencer_r2c.from_embeddings_with_bootstrapping(self.per_residue_embeddings,
                                                                                        self._test_targets_r2c,
                                                                                        iterations=10,
                                                                                        sample_size=3,
                                                                                        seed=42)
        r2c_dict = self.inferencer_r2c.from_embeddings(self.per_residue_embeddings, self._test_targets_r2c)
        for metric in r2c_dict["metrics"].keys():
            self.assertAlmostEqual(r2c_dict["metrics"][metric], r2c_dict_bootstrapping[metric]["mean"],
                                   delta=r2c_dict_bootstrapping[metric]["error"] * error_tolerance_factor)

        s2v_dict_bootstrapping = self.inferencer_s2v.from_embeddings_with_bootstrapping(self.per_sequence_embeddings,
                                                                                        self._test_targets_s2v,
                                                                                        iterations=10,
                                                                                        sample_size=2,
                                                                                        seed=42)
        s2v_dict = self.inferencer_s2v.from_embeddings(self.per_sequence_embeddings, self._test_targets_s2v)
        for metric in s2v_dict["metrics"].keys():
            self.assertAlmostEqual(s2v_dict["metrics"][metric], s2v_dict_bootstrapping[metric]["mean"],
                                   delta=s2v_dict_bootstrapping[metric]["error"] * error_tolerance_factor)

        # Check that a sample size and iteration of 1 work
        _ = self.inferencer_r2c.from_embeddings_with_bootstrapping(self.per_residue_embeddings,
                                                                   self._test_targets_r2c,
                                                                   iterations=1,
                                                                   sample_size=3,
                                                                   seed=42)
        _ = self.inferencer_s2v.from_embeddings_with_bootstrapping(self.per_sequence_embeddings,
                                                                   self._test_targets_s2v,
                                                                   iterations=10,
                                                                   sample_size=1,
                                                                   seed=42)

    def test_from_embeddings_with_dropout(self):
        r2c_dict = self.inferencer_r2c.from_embeddings_with_monte_carlo_dropout(self.per_residue_embeddings,
                                                                                n_forward_passes=10,
                                                                                confidence_level=0.05,
                                                                                seed=42)
        s2v_dict = self.inferencer_s2v.from_embeddings_with_monte_carlo_dropout(self.per_sequence_embeddings,
                                                                                n_forward_passes=10,
                                                                                confidence_level=0.05,
                                                                                seed=42)
        self.assertTrue(all([len(r2c_dict[seq]) == len(self._test_sequences[idx])
                             for idx, seq in enumerate(r2c_dict.keys())]))
        self.assertTrue(all([type(s2v_dict[key]["prediction"]) is float for key in s2v_dict.keys()]))
