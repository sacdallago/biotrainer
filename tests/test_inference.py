import tempfile
import unittest
import numpy as np

from typing import Dict, Union, List

from biotrainer.inference import Inferencer
from biotrainer.embedders import OneHotEncodingEmbedder
from biotrainer.protocols import Protocol


class InferencerTests(unittest.TestCase):
    _test_sequences = ["SEQVENCEPRTEI",
                       "SEQWENCE",
                       "PRTEIN"]
    _test_targets_r2c = ["CDEVVCCCDDDDD",
                         "DDDDDDDD",
                         "CDEVVV"]
    _test_targets_rs2c = ["TM", "TMSP", "Glob"]
    _test_targets_s2c = ["TM", "TMSP", "Glob"]
    _test_classes_r2c = ["C", "D", "E", "F", "V"]
    _test_classes_rs2c = ["Glob", "GlobSP", "TM", "TMSP"]
    _test_classes_s2c = ["Glob", "GlobSP", "TM", "TMSP"]
    _test_targets_s2v = [1, -1.212, 0.0]
    _test_targets_rs2v = [5, -1.212, 0.0]

    def setUp(self) -> None:
        self.inferencer_r2c, _ = Inferencer.create_from_out_file("test_input_files/test_models/r2c/out.yml")
        self.inferencer_rs2c, _ = Inferencer.create_from_out_file("test_input_files/test_models/rs2c/out.yml")
        self.inferencer_s2c, _ = Inferencer.create_from_out_file("test_input_files/test_models/s2c/out.yml")
        self.inferencer_s2v, _ = Inferencer.create_from_out_file("test_input_files/test_models/s2v/out.yml")
        self.inferencer_rs2v, _ = Inferencer.create_from_out_file("test_input_files/test_models/rs2v/out.yml")

        self.inferencer_list = [self.inferencer_r2c, self.inferencer_rs2c, self.inferencer_s2c, self.inferencer_s2v,
                                self.inferencer_rs2v]

        self.per_residue_embeddings, self.per_sequence_embeddings = self._embed()

    def _embed(self):
        embedder = OneHotEncodingEmbedder()
        per_residue_embeddings = list(embedder.embed_many(self._test_sequences))
        per_sequence_embeddings = [embedder.reduce_per_protein(embedding) for embedding in
                                   per_residue_embeddings]
        per_residue_embeddings_dict = {f"Seq{idx}": embed for idx, embed in enumerate(per_residue_embeddings)}
        per_sequence_embeddings_dict = {f"Seq{idx}": embed for idx, embed in enumerate(per_sequence_embeddings)}
        return per_residue_embeddings_dict, per_sequence_embeddings_dict

    def test_padding(self):
        max_seq_length = len(max(self._test_targets_r2c, key=len))
        for target in self._test_targets_r2c:
            self.assertTrue(len(self.inferencer_r2c._pad_target(self.inferencer_r2c._convert_class_str2int(target),
                                                                max_seq_length)) == max_seq_length,
                            "Padding did not enlarge target sequence to correct size!")
        for target in self._test_targets_s2v:
            self.assertTrue(self.inferencer_s2v._pad_target(target, max_seq_length) == target,
                            "Padding changed a non-list value!")

    def test_from_embeddings(self):
        r2c_dict = self.inferencer_r2c.from_embeddings(self.per_residue_embeddings)
        rs2c_dict = self.inferencer_rs2c.from_embeddings(self.per_residue_embeddings)
        s2c_dict = self.inferencer_s2c.from_embeddings(self.per_sequence_embeddings)
        s2v_dict = self.inferencer_s2v.from_embeddings(self.per_sequence_embeddings)
        rs2v_dict = self.inferencer_rs2v.from_embeddings(self.per_residue_embeddings)

        self.assertTrue("metrics" in r2c_dict.keys() and "metrics" in s2c_dict.keys() and "metrics" in s2v_dict.keys(),
                        "Missing metrics key!")
        self.assertTrue("mapped_predictions" in r2c_dict.keys() and "mapped_predictions" in s2v_dict.keys(),
                        "Missing predictions key!")

        self.assertTrue(all([pred_class in self._test_classes_r2c for pred_class in
                             "".join(r2c_dict["mapped_predictions"].values())]),
                        "Inferencer predicted a non-existing class!")
        self.assertTrue(all([pred_class in self._test_classes_rs2c
                             for pred_class in rs2c_dict["mapped_predictions"].values()]),
                        "Inferencer predicted a non-existing class (rs2c)!")
        self.assertTrue(all([pred_class in self._test_classes_s2c
                             for pred_class in s2c_dict["mapped_predictions"].values()]),
                        "Inferencer predicted a non-existing class (s2c)!")
        self.assertTrue(all([type(value) is float for value in s2v_dict["mapped_predictions"].values()]),
                        "Type of all sequence to value predictions is not float!")
        self.assertTrue(all([type(value) is float for value in rs2v_dict["mapped_predictions"].values()]),
                        "Type of all residues to value predictions is not float!")

    def test_from_embeddings_with_targets(self):
        error_tolerance = 0.01
        r2c_dict = self.inferencer_r2c.from_embeddings(self.per_residue_embeddings, self._test_targets_r2c)
        rs2c_dict = self.inferencer_rs2c.from_embeddings(self.per_residue_embeddings, self._test_targets_rs2c)
        s2c_dict = self.inferencer_s2c.from_embeddings(self.per_sequence_embeddings, self._test_targets_s2c)
        s2v_dict = self.inferencer_s2v.from_embeddings(self.per_sequence_embeddings, self._test_targets_s2v)
        rs2v_dict = self.inferencer_rs2v.from_embeddings(self.per_residue_embeddings, self._test_targets_rs2v)

        self.assertAlmostEqual(r2c_dict["metrics"]["loss"], 2.0788166522979736, delta=error_tolerance,
                               msg="Loss not as expected for r2c!")
        self.assertAlmostEqual(rs2c_dict["metrics"]["loss"], 1.5685200691223145, delta=error_tolerance,
                               msg="Loss not as expected for rs2c!")
        self.assertAlmostEqual(s2c_dict["metrics"]["loss"], 1.3706077337265015, delta=error_tolerance,
                               msg="Loss not as expected for s2c!")
        self.assertAlmostEqual(s2v_dict["metrics"]["loss"], 1.2870734930038452, delta=error_tolerance,
                               msg="Loss not as expected for s2v!")
        self.assertAlmostEqual(rs2v_dict["metrics"]["loss"], 46.71943283081055, delta=error_tolerance,
                               msg="Loss not as expected for rs2v!")

    def test_from_embeddings_with_bootstrapping(self):
        """
        Checks that metrics calculated with and without bootstrapping are within about the same range
        """
        error_tolerance_factor = 1.96
        # residue_to_class
        r2c_dict_bootstrapping = self.inferencer_r2c.from_embeddings_with_bootstrapping(self.per_residue_embeddings,
                                                                                        self._test_targets_r2c,
                                                                                        iterations=10,
                                                                                        sample_size=3,
                                                                                        seed=42)
        r2c_dict = self.inferencer_r2c.from_embeddings(self.per_residue_embeddings, self._test_targets_r2c)
        for metric in r2c_dict_bootstrapping.keys():
            self.assertAlmostEqual(r2c_dict["metrics"][metric], r2c_dict_bootstrapping[metric]["mean"],
                                   delta=r2c_dict_bootstrapping[metric]["error"] * error_tolerance_factor)

        # residues_to_class
        rs2c_dict_bootstrapping = self.inferencer_rs2c.from_embeddings_with_bootstrapping(self.per_residue_embeddings,
                                                                                          self._test_targets_rs2c,
                                                                                          iterations=6,
                                                                                          sample_size=2,
                                                                                          seed=42)
        rs2c_dict = self.inferencer_rs2c.from_embeddings(self.per_residue_embeddings, self._test_targets_rs2c)
        for metric in rs2c_dict_bootstrapping.keys():
            self.assertAlmostEqual(rs2c_dict["metrics"][metric], rs2c_dict_bootstrapping[metric]["mean"],
                                   delta=rs2c_dict_bootstrapping[metric]["error"] * error_tolerance_factor)

        # sequence_to_class
        s2c_dict_bootstrapping = self.inferencer_s2c.from_embeddings_with_bootstrapping(self.per_sequence_embeddings,
                                                                                        self._test_targets_s2c,
                                                                                        iterations=10,
                                                                                        sample_size=3,
                                                                                        seed=42)
        s2c_dict = self.inferencer_s2c.from_embeddings(self.per_sequence_embeddings, self._test_targets_s2c)
        for metric in s2c_dict_bootstrapping.keys():
            self.assertAlmostEqual(s2c_dict["metrics"][metric], s2c_dict_bootstrapping[metric]["mean"],
                                   delta=s2c_dict_bootstrapping[metric]["error"] * error_tolerance_factor)

        # sequence_to_value
        s2v_dict_bootstrapping = self.inferencer_s2v.from_embeddings_with_bootstrapping(self.per_sequence_embeddings,
                                                                                        self._test_targets_s2v,
                                                                                        iterations=10,
                                                                                        sample_size=2,
                                                                                        seed=42)
        s2v_dict = self.inferencer_s2v.from_embeddings(self.per_sequence_embeddings, self._test_targets_s2v)
        for metric in s2v_dict_bootstrapping.keys():
            self.assertAlmostEqual(s2v_dict["metrics"][metric], s2v_dict_bootstrapping[metric]["mean"],
                                   delta=s2v_dict_bootstrapping[metric]["error"] * error_tolerance_factor)

        # residues_to_value
        rs2v_dict_bootstrapping = self.inferencer_rs2v.from_embeddings_with_bootstrapping(self.per_residue_embeddings,
                                                                                          self._test_targets_rs2v,
                                                                                          iterations=10,
                                                                                          sample_size=2,
                                                                                          seed=42)
        rs2v_dict = self.inferencer_rs2v.from_embeddings(self.per_residue_embeddings, self._test_targets_rs2v)
        for metric in rs2v_dict_bootstrapping.keys():
            self.assertAlmostEqual(rs2v_dict["metrics"][metric], rs2v_dict_bootstrapping[metric]["mean"],
                                   delta=rs2v_dict_bootstrapping[metric]["error"] * error_tolerance_factor)

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
        # Check that larger sample sizes than provided sequence number work
        _ = self.inferencer_s2v.from_embeddings_with_bootstrapping(self.per_sequence_embeddings,
                                                                   self._test_targets_s2v,
                                                                   iterations=5,
                                                                   sample_size=100,
                                                                   seed=42)

    def test_from_embeddings_with_dropout(self):
        r2c_dict = self.inferencer_r2c.from_embeddings_with_monte_carlo_dropout(self.per_residue_embeddings,
                                                                                n_forward_passes=10,
                                                                                confidence_level=0.05,
                                                                                seed=42)
        rs2c_dict = self.inferencer_rs2c.from_embeddings_with_monte_carlo_dropout(self.per_residue_embeddings,
                                                                                  n_forward_passes=15,
                                                                                  confidence_level=0.5,
                                                                                  seed=42)
        s2c_dict = self.inferencer_s2c.from_embeddings_with_monte_carlo_dropout(self.per_sequence_embeddings,
                                                                                n_forward_passes=1,
                                                                                confidence_level=0.01,
                                                                                seed=42)
        s2v_dict = self.inferencer_s2v.from_embeddings_with_monte_carlo_dropout(self.per_sequence_embeddings,
                                                                                n_forward_passes=10,
                                                                                confidence_level=0.05,
                                                                                seed=42)
        rs2v_dict = self.inferencer_rs2v.from_embeddings_with_monte_carlo_dropout(self.per_residue_embeddings,
                                                                                  n_forward_passes=10,
                                                                                  confidence_level=0.05,
                                                                                  seed=42)
        self.assertTrue(all([len(r2c_dict[seq]) == len(self._test_sequences[idx])
                             for idx, seq in enumerate(r2c_dict.keys())]), msg="Missing predictions for r2c!")
        self.assertTrue("mcd_mean" in list(rs2c_dict.values())[0].keys(), "Missing mcd_mean value for rs2c!")
        self.assertTrue("mcd_lower_bound" in list(s2c_dict.values())[0].keys(),
                        "Missing mcd_lower_bound value for s2c!")
        self.assertTrue("mcd_upper_bound" in list(s2c_dict.values())[0].keys(),
                        "Missing mcd_upper_bound value for s2c!")
        self.assertTrue(all([type(s2v_dict[key]["prediction"]) is float for key in s2v_dict.keys()]),
                        msg="Regression prediction is not float!")
        self.assertTrue(all([type(rs2v_dict[key]["prediction"]) is float for key in rs2v_dict.keys()]),
                        msg="Regression prediction is not float!")

    @staticmethod
    def _compare_predictions(onnx_results: Dict[str, Union[float, List, List[List]]],
                             inferencer_results: Dict[str, Union[float, List, List[List]]]) -> List[str]:
        """
        Compare ONNX and Inferencer predictions, handling both per-sequence and per-residue cases for classification
        and regression.

        :param onnx_results: Dictionary of ONNX predictions
        :param inferencer_results: Dictionary of Inferencer predictions
        :return: List of error messages, empty if no errors
        """
        # TODO The error tolerance is very large at the moment because of padding-related differences in the last
        # TODO positions of per-residue embeddings predictions
        error_tolerance = 0.2

        error_messages = []
        for key in onnx_results.keys():
            onnx_pred = np.array(onnx_results[key])
            inf_pred = np.array(inferencer_results[key])

            if onnx_pred.shape != inf_pred.shape:
                error_messages.append(
                    f"Shape mismatch for key {key}: ONNX {onnx_pred.shape} vs Inferencer {inf_pred.shape}")
                continue

            diff = np.abs(onnx_pred - inf_pred)
            max_diff = np.max(diff)

            if max_diff > error_tolerance:
                error_location = np.unravel_index(np.argmax(diff), diff.shape)
                error_messages.append(
                    f"Prediction mismatch for key {key} at index {error_location}: "
                    f"ONNX {onnx_pred[error_location]:.6f} vs Inferencer {inf_pred[error_location]:.6f}, "
                    f"difference {max_diff:.6f}"
                )

        return error_messages

    def test_onnx_conversion(self):
        embedding_dimension_one_hot = 21

        for inferencer in self.inferencer_list:
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                # Convert
                converted_file_paths = inferencer.convert_to_onnx(embedding_dimension_one_hot, tmp_dir_name)
                self.assertTrue(len(converted_file_paths) == 1)

                # Load converted model and make predictions
                model_path = converted_file_paths[0]
                embeddings = self.per_sequence_embeddings \
                    if inferencer.protocol in Protocol.using_per_sequence_embeddings() else self.per_residue_embeddings

                onnx_result_dict = Inferencer.from_onnx_with_embeddings(model_path=model_path,
                                                                        embeddings=embeddings,
                                                                        protocol=inferencer.protocol)

                # Compare to inferencer predictions
                inferencer_result_dict = inferencer.from_embeddings(embeddings=embeddings,
                                                                    include_probabilities=True)
                inferencer_result_dict_prob = inferencer_result_dict["mapped_probabilities"]

                prediction_errors = self._compare_predictions(onnx_results=onnx_result_dict,
                                                              inferencer_results=inferencer_result_dict_prob)
                if len(prediction_errors) > 0:
                    print(prediction_errors)
                self.assertTrue(len(prediction_errors) == 0)
