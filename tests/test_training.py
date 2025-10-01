import unittest
import tempfile

from copy import deepcopy
from biotrainer.utilities.cli import train

s2c_config = {'protocol': 'sequence_to_class',
              'model_choice': 'FNN',
              'embedder_name': 'one_hot_encoding',
              'input_file': "test_input_files/scl_subset/scl_rand.fasta"}


class TrainingTests(unittest.TestCase):
    error_tolerance = 0.01

    @staticmethod
    def _get_best_training_loss_from_result(result_dict):
        return result_dict['training_results']['hold_out']['best_training_epoch_metrics']['training']['loss']

    def test_train_with_and_without_class_weights(self):
        """ Checks that training with and without class weights results in different models"""
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2c_config)
            config["output_dir"] = tmp_dir_name
            config["use_class_weights"] = True
            result_with_weights = train(config=config)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2c_config)
            config["output_dir"] = tmp_dir_name
            config["use_class_weights"] = False
            result_without_weights = train(config=config)
        self.assertTrue(result_with_weights['config']['use_class_weights'])
        self.assertFalse(result_without_weights['config']['use_class_weights'])

        self.assertNotEqual(self._get_best_training_loss_from_result(result_with_weights),
                            self._get_best_training_loss_from_result(result_without_weights))