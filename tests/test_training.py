import torch
import unittest
import tempfile
import numpy as np

from copy import deepcopy
from biotrainer_core.data_classes import SequenceData, BiotrainerModelResult

from biotrainer.training import BiotrainerModel
from biotrainer.training.utilities import FeatureScaler

s2c_config = {'protocol': 'sequence_to_class',
              'model_choice': 'FNN',
              'embedder_name': 'one_hot_encoding',
              'input_file': "test_input_files/scl_subset/scl_rand.fasta"}

rs2c_config = {'protocol': 'residues_to_class',
               'model_choice': 'LightAttention',
               'embedder_name': 'one_hot_encoding',
               'input_file': "test_input_files/scl_subset/scl_rand.fasta"}

s2v_random_config = {
    "protocol": "sequence_to_value",
    "model_choice": "FNN",
    "embedder_name": "random_embedder",
    "num_epochs": 2,
    "seed": 1510,
}

s2v_input_data = [
    SequenceData(
        seq_id=f"Seq{i}", seq=seq, attributes={"TARGET": str(target), "SET": split}
    )
    for i, (seq, target, split) in enumerate(
        [
            ("MMALSLALM", 0.1, "train"),
            ("PRTEIN", 0.4, "train"),
            ("PRT", 0.9, "train"),
            ("MKTAYIAK", 0.3, "train"),
            ("SEQWENCE", 0.7, "val"),
            ("PRTE", 0.2, "val"),
            ("MMALSM", 0.5, "test"),
            ("PRSEQ", 0.8, "test"),
        ]
    )
]


class TrainingTests(unittest.TestCase):
    error_tolerance = 0.01

    @staticmethod
    def _get_best_training_loss_from_result(result: BiotrainerModelResult):
        return result.training_results['hold_out'].best_epoch_metrics.training['loss']

    def test_train_with_and_without_class_weights(self):
        """ Checks that training with and without class weights results in different models"""
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2c_config)
            config["output_dir"] = tmp_dir_name
            config["use_class_weights"] = True
            result_with_weights = BiotrainerModel().train(config=config)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2c_config)
            config["output_dir"] = tmp_dir_name
            config["use_class_weights"] = False
            result_without_weights = BiotrainerModel().train(config=config)
        self.assertTrue(result_with_weights.config['use_class_weights'])
        self.assertFalse(result_without_weights.config['use_class_weights'])

        self.assertNotEqual(self._get_best_training_loss_from_result(result_with_weights),
                            self._get_best_training_loss_from_result(result_without_weights))

    def test_train_with_and_without_feature_scaling_per_sequence(self):
        """ Checks that training with and without scaling results in different models"""
        results = []
        for method in FeatureScaler.methods().keys():
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                config = deepcopy(s2c_config)
                config["output_dir"] = tmp_dir_name
                config["scaling_method"] = str(method)
                result = BiotrainerModel().train(config=config)
                self.assertEqual(result.config['scaling_method'], method,
                                 f"Scaling method {method} not saved in output config!")
                results.append(self._get_best_training_loss_from_result(result))
        # Check all results are different
        self.assertTrue(len(set(results)) == len(results))

    def test_direct_input_data(self):
        input_data = [
            SequenceData(seq_id="Seq1", seq="MMALSLALM", attributes={"TARGET": "Membrane", "SET": "train"}),
            SequenceData(seq_id="Seq2", seq="PRTEIN", attributes={"TARGET": "Membrane", "SET": "train"}),
            SequenceData(seq_id="Seq3", seq="PRT", attributes={"TARGET": "Soluble", "SET": "train"}),
            SequenceData(seq_id="Seq4", seq="SEQWENCE", attributes={"TARGET": "Membrane", "SET": "val"}),
            SequenceData(seq_id="Seq5", seq="PRTE", attributes={"TARGET": "Soluble", "SET": "val"}),
            SequenceData(seq_id="Seq6", seq="MMALSM", attributes={"TARGET": "Membrane", "SET": "test"}),
            SequenceData(seq_id="Seq7", seq="PRSEQ", attributes={"TARGET": "Soluble", "SET": "test"}),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2c_config)
            config.pop("input_file")
            result = BiotrainerModel().train(config=config, input_data=input_data)
            self.assertEqual(result.config['input_data'], len(input_data))

    def test_direct_input_data_with_embeddings(self):
        # Success
        input_data = [
            SequenceData(seq_id="Seq1", seq="MMALSLALM", attributes={"TARGET": "Membrane", "SET": "train"},
                                     embedding=[1, 2, 3]),
            SequenceData(seq_id="Seq2", seq="PRTEIN", attributes={"TARGET": "Membrane", "SET": "train"},
                                     embedding=[4, 5, 6]),
            SequenceData(seq_id="Seq3", seq="PRT", attributes={"TARGET": "Soluble", "SET": "train"},
                                     embedding=[7, 8, 9]),
            SequenceData(seq_id="Seq4", seq="SEQWENCE", attributes={"TARGET": "Membrane", "SET": "val"},
                                     embedding=[10, 11, 12]),
            SequenceData(seq_id="Seq5", seq="PRTE", attributes={"TARGET": "Soluble", "SET": "val"},
                                     embedding=[13, 14, 15]),
            SequenceData(seq_id="Seq6", seq="MMALSM", attributes={"TARGET": "Membrane", "SET": "test"},
                                     embedding=torch.tensor([16, 17, 18])),
            SequenceData(seq_id="Seq7", seq="PRSEQ", attributes={"TARGET": "Soluble", "SET": "test"},
                                     embedding=np.array([19, 20, 21])),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2c_config)
            config.pop("input_file")
            result = BiotrainerModel().train(config=config, input_data=input_data)
            self.assertEqual(result.config['input_data'], len(input_data))

        # Errors
        # Missing embedding
        input_data[0] = SequenceData(seq_id="Seq1", seq="MMALSLALM",
                                                 attributes={"TARGET": "Membrane", "SET": "train"})

        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                config = deepcopy(s2c_config)
                config.pop("input_file")
                result = BiotrainerModel().train(config=config, input_data=input_data)

    def test_training_embeds_with_the_config_seed(self):
        """Two trainings with the same seed embed alike, so they reach the same training loss"""
        losses = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                config = deepcopy(s2v_random_config)
                config["output_dir"] = tmp_dir_name
                result = BiotrainerModel().train(
                    config=config, input_data=deepcopy(s2v_input_data)
                )
                losses.append(self._get_best_training_loss_from_result(result))
        self.assertEqual(losses[0], losses[1])

    def test_prediction_embeds_with_the_training_seed(self):
        """Predicting twice with a random-embedder model embeds alike, so the predictions agree"""
        new_sequences = [
            SequenceData(seq_id="New1", seq="MKTAYIAKQR"),
            SequenceData(seq_id="New2", seq="SEQVENCE"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2v_random_config)
            config["output_dir"] = tmp_dir_name
            model = BiotrainerModel()
            model.train(config=config, input_data=deepcopy(s2v_input_data))
            first = model.predict(deepcopy(new_sequences)).predictions
            second = model.predict(deepcopy(new_sequences)).predictions
        self.assertEqual(first, second)
