import unittest
import tempfile

from copy import deepcopy

from biotrainer.input_files import BiotrainerSequenceRecord
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

    def test_direct_input_data(self):
        input_data = [
            BiotrainerSequenceRecord(seq_id="Seq1", seq="MMALSLALM", attributes={"TARGET": "Membrane", "SET": "train"}),
            BiotrainerSequenceRecord(seq_id="Seq2", seq="PRTEIN", attributes={"TARGET": "Membrane", "SET": "train"}),
            BiotrainerSequenceRecord(seq_id="Seq3", seq="PRT", attributes={"TARGET": "Soluble", "SET": "train"}),
            BiotrainerSequenceRecord(seq_id="Seq4", seq="SEQWENCE", attributes={"TARGET": "Membrane", "SET": "val"}),
            BiotrainerSequenceRecord(seq_id="Seq5", seq="PRTE", attributes={"TARGET": "Soluble", "SET": "val"}),
            BiotrainerSequenceRecord(seq_id="Seq6", seq="MMALSM", attributes={"TARGET": "Membrane", "SET": "test"}),
            BiotrainerSequenceRecord(seq_id="Seq7", seq="PRSEQ", attributes={"TARGET": "Soluble", "SET": "test"}),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2c_config)
            config.pop("input_file")
            config["input_data"] = input_data
            result = train(config=config)
            self.assertEqual(result['config']['input_data'], len(input_data))

