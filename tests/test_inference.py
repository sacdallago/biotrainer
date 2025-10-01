import tempfile
import unittest
import numpy as np
import torch._dynamo

from sys import platform
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
    _test_targets_r2v = [[0.05, 1.0, 2, 1.1, -1.2, 1.11, 5.4, 33, 0.12, -1.0, 0.0, 12, 14.131],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [-1, 0, -0, 5.5, 1.01, 12.0001]]
    _test_targets_rs2c = ["TM", "TMSP", "Glob"]
    _test_targets_s2c = ["TM", "TMSP", "Glob"]
    _test_classes_r2c = ["C", "D", "E", "F", "V"]
    _test_classes_rs2c = ["Glob", "GlobSP", "TM", "TMSP"]
    _test_classes_s2c = ["Glob", "GlobSP", "TM", "TMSP"]
    _test_targets_s2v = [1, -1.212, 0.0]
    _test_targets_rs2v = [5, -1.212, 0.0]

    error_tolerance = 0.02
    error_tolerance_factor = 1.96

    def setUp(self) -> None:
        # Needed for cross-platform compatibility of Windows and Linux models
        torch._dynamo.config.suppress_errors = True

        self.inferencer_r2c, _ = Inferencer.create_from_out_file("test_input_files/test_models/r2c/out.yml")
        self.inferencer_r2v, _ = Inferencer.create_from_out_file("test_input_files/test_models/r2v/out.yml")
        self.inferencer_rs2c, _ = Inferencer.create_from_out_file("test_input_files/test_models/rs2c/out.yml")
        self.inferencer_s2c, _ = Inferencer.create_from_out_file("test_input_files/test_models/s2c/out.yml")
        self.inferencer_s2v, _ = Inferencer.create_from_out_file("test_input_files/test_models/s2v/out.yml")
        self.inferencer_rs2v, _ = Inferencer.create_from_out_file("test_input_files/test_models/rs2v/out.yml")

        self.inferencer_list = [self.inferencer_r2c, self.inferencer_rs2c, self.inferencer_s2c, self.inferencer_s2v,
                                self.inferencer_rs2v]

        self.per_residue_embeddings, self.per_sequence_embeddings = self._embed()

    @classmethod
    def tearDownClass(cls) -> None:
        # Undo error suppressing in dynamo
        torch._dynamo.config.suppress_errors = False

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
            self.assertTrue(len(self.inferencer_r2c._pad_tensor(protocol=self.inferencer_r2c.protocol,
                                                                target=self.inferencer_r2c._convert_class_str2int(
                                                                    target),
                                                                length_to_pad=max_seq_length,
                                                                device="cpu")) == max_seq_length,
                            "Padding did not enlarge target sequence to correct size!")
        for target in self._test_targets_s2v:
            self.assertTrue(self.inferencer_s2v._pad_tensor(protocol=self.inferencer_s2v.protocol, target=target,
                                                            length_to_pad=max_seq_length, device="cpu") == target,
                            "Padding changed a non-list value!")

    def test_from_embeddings(self):
        r2c_dict = self.inferencer_r2c.from_embeddings(self.per_residue_embeddings)
        r2v_dict = self.inferencer_r2v.from_embeddings(self.per_residue_embeddings)
        rs2c_dict = self.inferencer_rs2c.from_embeddings(self.per_residue_embeddings)
        s2c_dict = self.inferencer_s2c.from_embeddings(self.per_sequence_embeddings)
        s2v_dict = self.inferencer_s2v.from_embeddings(self.per_sequence_embeddings)
        rs2v_dict = self.inferencer_rs2v.from_embeddings(self.per_residue_embeddings)

        self.assertTrue(
            "metrics" in r2c_dict.keys() and "metrics" in r2v_dict.keys() and
            "metrics" in s2c_dict.keys() and "metrics" in s2v_dict.keys(),
            "Missing metrics key!")
        self.assertTrue("mapped_predictions" in r2c_dict.keys() and "mapped_predictions" in s2v_dict.keys(),
                        "Missing predictions key!")

        self.assertTrue(all([pred_class in self._test_classes_r2c for pred_class in
                             "".join(r2c_dict["mapped_predictions"].values())]),
                        "Inferencer predicted a non-existing class!")
        self.assertTrue(all([type(value) for value in
                             [v for v in (vs for vs in r2v_dict["mapped_predictions"].values())]]),
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
        r2c_dict = self.inferencer_r2c.from_embeddings(self.per_residue_embeddings, self._test_targets_r2c)
        r2v_dict = self.inferencer_r2v.from_embeddings(self.per_residue_embeddings, self._test_targets_r2v)
        rs2c_dict = self.inferencer_rs2c.from_embeddings(self.per_residue_embeddings, self._test_targets_rs2c)
        s2c_dict = self.inferencer_s2c.from_embeddings(self.per_sequence_embeddings, self._test_targets_s2c)
        s2v_dict = self.inferencer_s2v.from_embeddings(self.per_sequence_embeddings, self._test_targets_s2v)
        rs2v_dict = self.inferencer_rs2v.from_embeddings(self.per_residue_embeddings, self._test_targets_rs2v)

        self.assertAlmostEqual(r2c_dict["metrics"]["loss"], 2.13, delta=self.error_tolerance,
                               msg="Loss not as expected for r2c!")
        self.assertAlmostEqual(r2v_dict["metrics"]["loss"], 66.42302047378135, delta=self.error_tolerance,
                               msg="Loss not as expected for r2v!")
        self.assertAlmostEqual(rs2c_dict["metrics"]["loss"], 1.7416267395019531, delta=self.error_tolerance,
                               msg="Loss not as expected for rs2c!")
        self.assertAlmostEqual(s2c_dict["metrics"]["loss"], 1.3706077337265015, delta=self.error_tolerance,
                               msg="Loss not as expected for s2c!")
        self.assertAlmostEqual(s2v_dict["metrics"]["loss"], 1.2276635438517511, delta=self.error_tolerance,
                               msg="Loss not as expected for s2v!")
        self.assertAlmostEqual(rs2v_dict["metrics"]["loss"], 107.96429166959688, delta=self.error_tolerance,
                               msg="Loss not as expected for rs2v!")

    def test_from_embeddings_with_bootstrapping(self):
        """
        Checks that metrics calculated with and without bootstrapping are within about the same range
        """

        def _get_error_range(bootstrapping_dict):
            return (bootstrapping_dict["upper"] - bootstrapping_dict["lower"]) * 0.5 * self.error_tolerance_factor

        # residue_to_class
        r2c_dict_bootstrapping = self.inferencer_r2c.from_embeddings_with_bootstrapping(self.per_residue_embeddings,
                                                                                        self._test_targets_r2c,
                                                                                        iterations=10,
                                                                                        sample_size=3,
                                                                                        seed=42)
        r2c_dict = self.inferencer_r2c.from_embeddings(self.per_residue_embeddings, self._test_targets_r2c)
        for metric in r2c_dict_bootstrapping.keys():
            self.assertAlmostEqual(r2c_dict["metrics"][metric], r2c_dict_bootstrapping[metric]["mean"],
                                   delta=_get_error_range(r2c_dict_bootstrapping[metric]))

        # residues_to_class
        rs2c_dict_bootstrapping = self.inferencer_rs2c.from_embeddings_with_bootstrapping(self.per_residue_embeddings,
                                                                                          self._test_targets_rs2c,
                                                                                          iterations=6,
                                                                                          sample_size=2,
                                                                                          seed=42)
        rs2c_dict = self.inferencer_rs2c.from_embeddings(self.per_residue_embeddings, self._test_targets_rs2c)
        for metric in rs2c_dict_bootstrapping.keys():
            self.assertAlmostEqual(rs2c_dict["metrics"][metric], rs2c_dict_bootstrapping[metric]["mean"],
                                   delta=_get_error_range(rs2c_dict_bootstrapping[metric]))

        # sequence_to_class
        s2c_dict_bootstrapping = self.inferencer_s2c.from_embeddings_with_bootstrapping(self.per_sequence_embeddings,
                                                                                        self._test_targets_s2c,
                                                                                        iterations=10,
                                                                                        sample_size=3,
                                                                                        seed=42)
        s2c_dict = self.inferencer_s2c.from_embeddings(self.per_sequence_embeddings, self._test_targets_s2c)
        for metric in s2c_dict_bootstrapping.keys():
            self.assertAlmostEqual(s2c_dict["metrics"][metric], s2c_dict_bootstrapping[metric]["mean"],
                                   delta=_get_error_range(s2c_dict_bootstrapping[metric]))

        # sequence_to_value
        s2v_dict_bootstrapping = self.inferencer_s2v.from_embeddings_with_bootstrapping(self.per_sequence_embeddings,
                                                                                        self._test_targets_s2v,
                                                                                        iterations=10,
                                                                                        sample_size=2,
                                                                                        seed=42)
        s2v_dict = self.inferencer_s2v.from_embeddings(self.per_sequence_embeddings, self._test_targets_s2v)
        for metric in s2v_dict_bootstrapping.keys():
            self.assertAlmostEqual(s2v_dict["metrics"][metric], s2v_dict_bootstrapping[metric]["mean"],
                                   delta=_get_error_range(s2v_dict_bootstrapping[metric]))

        # residues_to_value
        rs2v_dict_bootstrapping = self.inferencer_rs2v.from_embeddings_with_bootstrapping(self.per_residue_embeddings,
                                                                                          self._test_targets_rs2v,
                                                                                          iterations=10,
                                                                                          sample_size=2,
                                                                                          seed=42)
        rs2v_dict = self.inferencer_rs2v.from_embeddings(self.per_residue_embeddings, self._test_targets_rs2v)
        for metric in rs2v_dict_bootstrapping.keys():
            self.assertAlmostEqual(rs2v_dict["metrics"][metric], rs2v_dict_bootstrapping[metric]["mean"],
                                   delta=_get_error_range(rs2v_dict_bootstrapping[metric]))

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
        # Test with different parameters
        r2c_dict = self.inferencer_r2c.from_embeddings_with_monte_carlo_dropout(
            self.per_residue_embeddings,
            n_forward_passes=10,
            confidence_level=0.05,
            seed=42
        )
        r2v_dict = self.inferencer_r2v.from_embeddings_with_monte_carlo_dropout(
            self.per_residue_embeddings,
            n_forward_passes=10,
            confidence_level=0.05,
            seed=42
        )
        rs2c_dict = self.inferencer_rs2c.from_embeddings_with_monte_carlo_dropout(
            self.per_residue_embeddings,
            n_forward_passes=15,
            confidence_level=0.5,
            seed=42
        )
        s2c_dict = self.inferencer_s2c.from_embeddings_with_monte_carlo_dropout(
            self.per_sequence_embeddings,
            n_forward_passes=2,
            confidence_level=0.01,
            seed=42
        )
        s2v_dict = self.inferencer_s2v.from_embeddings_with_monte_carlo_dropout(
            self.per_sequence_embeddings,
            n_forward_passes=10,
            confidence_level=0.05,
            seed=42
        )
        rs2v_dict = self.inferencer_rs2v.from_embeddings_with_monte_carlo_dropout(
            self.per_residue_embeddings,
            n_forward_passes=10,
            confidence_level=0.05,
            seed=42
        )

        # Basic structure tests
        self.assertTrue(all([len(r2c_dict[seq]) == len(self._test_sequences[idx])
                             for idx, seq in enumerate(r2c_dict.keys())]),
                        msg="Missing predictions for r2c!")

        self.assertTrue(all([len(r2v_dict[seq]) == len(self._test_sequences[idx])
                             for idx, seq in enumerate(r2v_dict.keys())]),
                        msg="Missing predictions for r2v!")

        # Check required keys exist
        required_keys = ["prediction", "all_predictions", "mcd_mean",
                         "mcd_lower_bound", "mcd_upper_bound"]

        for dict_to_check in [rs2c_dict, s2c_dict, s2v_dict, rs2v_dict]:
            for key in required_keys:
                self.assertTrue(key in list(dict_to_check.values())[0].keys(),
                                f"Missing {key} in predictions!")

        # Check confidence bounds
        for dict_to_check in [rs2c_dict, s2c_dict, s2v_dict, rs2v_dict]:
            for pred in dict_to_check.values():
                self.assertTrue(np.all((pred["mcd_lower_bound"] <= pred["mcd_mean"]).numpy()),
                                f"Lower bound greater than mean ({pred})!")
                self.assertTrue(np.all((pred["mcd_upper_bound"] >= pred["mcd_mean"]).numpy()),
                                f"Upper bound less than mean ({pred})!")

        # Check number of forward passes
        self.assertEqual(len(s2c_dict["Seq0"]["all_predictions"]), 2,
                         "Incorrect number of forward passes for s2c!")
        self.assertEqual(len(r2c_dict["Seq0"][0]["all_predictions"]), 10,
                         "Incorrect number of forward passes for r2c!")

        # Check prediction types
        self.assertTrue(all([type(s2v_dict[key]["prediction"]) is float for key in s2v_dict.keys()]),
                        msg="Regression prediction is not float!")
        self.assertTrue(all([type(rs2v_dict[key]["prediction"]) is float for key in rs2v_dict.keys()]),
                        msg="Regression prediction is not float!")

        # Test reproducibility with same seed
        s2v_dict_repeat = self.inferencer_s2v.from_embeddings_with_monte_carlo_dropout(
            self.per_sequence_embeddings,
            n_forward_passes=10,
            confidence_level=0.05,
            seed=42
        )
        for key in s2v_dict.keys():
            np.testing.assert_array_almost_equal(
                s2v_dict[key]["all_predictions"],
                s2v_dict_repeat[key]["all_predictions"],
                decimal=5,
                err_msg="MC Dropout not reproducible with same seed!"
            )

        # Test different seeds give different results
        s2v_dict_different_seed = self.inferencer_s2v.from_embeddings_with_monte_carlo_dropout(
            self.per_sequence_embeddings,
            n_forward_passes=10,
            confidence_level=0.05,
            seed=24
        )
        for key in s2v_dict.keys():
            self.assertFalse(
                np.array_equal(s2v_dict[key]["all_predictions"],
                               s2v_dict_different_seed[key]["all_predictions"]),
                "Different seeds produce identical results!"
            )

        # Test invalid parameters
        with self.assertRaises(Exception):
            self.inferencer_s2v.from_embeddings_with_monte_carlo_dropout(
                self.per_sequence_embeddings,
                n_forward_passes=0,  # Invalid number of passes
                confidence_level=0.05,
                seed=42
            )

        with self.assertRaises(Exception):
            self.inferencer_r2c.from_embeddings_with_monte_carlo_dropout(
                self.per_sequence_embeddings,
                n_forward_passes=1,  # Invalid number of passes
                confidence_level=0.05,
                seed=42
            )

        with self.assertRaises(Exception):
            self.inferencer_s2v.from_embeddings_with_monte_carlo_dropout(
                self.per_sequence_embeddings,
                n_forward_passes=10,
                confidence_level=1.5,  # Invalid confidence level
                seed=42
            )

    def _compare_predictions(self, preds: Dict[str, Union[float, List, List[List]]],
                             other_preds: Dict[str, Union[float, List, List[List]]]) -> List[str]:
        """
        Compare two predictions, handling both per-sequence and per-residue cases for classification
        and regression.

        :param preds: Dictionary of predictions
        :param other_preds: Dictionary of other predictions to compare
        :return: List of error messages, empty if no errors
        """
        error_messages = []
        for key in preds.keys():
            pred = np.array(preds[key])
            pred_other = np.array(other_preds[key])

            if pred.shape != pred_other.shape:
                error_messages.append(
                    f"Shape mismatch for key {key}: Predictions {pred.shape} vs other predictions {pred_other.shape}")
                continue

            diff = np.abs(pred - pred_other)
            max_diff = np.max(diff)

            if max_diff > self.error_tolerance:
                error_location = np.unravel_index(np.argmax(diff), diff.shape)
                error_messages.append(
                    f"Prediction mismatch for key {key} at index {error_location}: "
                    f"Predictions {pred[error_location]:.6f} vs other predictions {pred_other[error_location]:.6f}, "
                    f"difference {max_diff:.6f}"
                )

        return error_messages

    def test_onnx_conversion(self):
        return  # Don't run this test anymore, because of ONNX stability issues in the CI pipeline

        for inferencer in self.inferencer_list:
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                # Convert
                converted_file_paths = inferencer.convert_to_onnx(tmp_dir_name)
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

                prediction_errors = self._compare_predictions(preds=onnx_result_dict,
                                                              other_preds=inferencer_result_dict_prob)
                if len(prediction_errors) > 0:
                    print(prediction_errors)
                self.assertTrue(len(prediction_errors) == 0)

    def test_single_vs_batch_prediction(self):
        for inferencer in self.inferencer_list:
            embeddings = self.per_sequence_embeddings \
                if inferencer.protocol in Protocol.using_per_sequence_embeddings() else self.per_residue_embeddings

            pred_dict_batch = inferencer.from_embeddings(embeddings=embeddings, include_probabilities=True)

            pred_dict_single = {}
            for seq_id, emb in embeddings.items():
                pred_dict_single[seq_id] = inferencer.from_embeddings({seq_id: emb},
                                                                      include_probabilities=True)[
                    "mapped_probabilities"][seq_id]

            # Single predictions and batch predictions should not significantly differ
            prediction_errors = self._compare_predictions(pred_dict_batch["mapped_probabilities"], pred_dict_single)
            if len(prediction_errors) > 0:
                print(prediction_errors)
            self.assertTrue(len(prediction_errors) == 0)
