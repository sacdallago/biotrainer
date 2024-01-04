import unittest

from pathlib import Path
from copy import deepcopy

from biotrainer.config import Configurator, ConfigurationException

configurations = {
    "minimal": {
        "sequence_file": "test_input_files/r2c/sequences.fasta",
        "protocol": "sequence_to_class",
    },
    "prohibited": {
        "sequence_file": "test_input_files/r2c/sequences.fasta",
        "labels_file": "test_input_files/r2c/labels.fasta",
        "protocol": "sequence_to_class"
    },
    "required": {
        # Missing sequence file
        "labels_file": "test_input_files/r2c/labels.fasta",
        "protocol": "residue_to_class"
    },
    "download": {
        "sequence_file": "https://example.com/sequences.fasta",
        "protocol": "sequence_to_class",
        "model_choice": "FNN",
    },
    "download_prohibited": {
        "sequence_file": "test_input_files/r2c/sequences.fasta",
        "protocol": "sequence_to_class",
        "model_choice": "FNN",
        "embedder_name": "https://example.com/payload.py"
    },
    "auto_resume_pretrained_model_mutual_exclusive": {
        "sequence_file": "test_input_files/r2c/sequences.fasta",
        "protocol": "sequence_to_class",
        "auto_resume": True,
        "pretrained_model": "placeholder.pt"
    },
    "embeddings_file_embedder_name_mutual_exclusive": {
        "sequence_file": "test_input_files/r2c/sequences.fasta",
        "protocol": "sequence_to_class",
        "embeddings_file": "placeholder.h5",
        "embedder_name": "one_hot_encoding"
    },
    "local_custom_embedder": {
        "sequence_file": "test_input_files/r2c/sequences.fasta",
        "protocol": "sequence_to_class",
        "embedder_name": "../examples/custom_embedder/ankh/ankh_embedder.py"
    },
    "file_paths": {
        "sequence_file": "test_input_files/r2c/sequences.fasta",
        "labels_file": "test_input_files/r2c/labels.fasta",
        "mask_file": "test_input_files/r2c/mask.fasta",
        "embedder_name": "../examples/custom_embedder/esm2/esm2_embedder.py",
        "output_dir": "test_output",
        "protocol": "residue_to_class"
    },
    "multiple_values": {
        "sequence_file": "test_input_files/r2c/sequences.fasta",
        "protocol": "sequence_to_class",
        "model_choice": ["FNN", "DeeperFNN"],
        "num_epochs": [100, 200],
        "learning_rate": [1e-3, 1e-4],
        "batch_size": [64, 128],
        "epsilon": [1e-4, 1e-3],
        "patience": [10, 20],
        "use_class_weights": [True, False]
    },
    "hold_out": {
        "cross_validation_config": {
            "method": "hold_out",
            "choose_by": "loss"
        }
    },
    "k_fold": {
        "cross_validation_config": {
            "method": "k_fold",
            "k": 3,
            "stratified": True,
            "repeat": 1
        }
    },
    "nested_k_fold": {
        "cross_validation_config": {
            "method": "k_fold",
            "repeat": 2,
            "k": 4,
            "nested": True,
            "nested_k": 3,
            "search_method": "random_search",
            "n_max_evaluations_random": 5
        }
    },
    "leave_p_out": {
        "cross_validation_config": {
            "method": "leave_p_out",
            "p": 5,
        }
    }
}


class ConfigurationVerificationTests(unittest.TestCase):

    def test_minimal_configuration(self):
        configurator = Configurator.from_config_dict(configurations["minimal"])
        self.assertTrue(configurator.get_verified_config(), "Minimal config does not work!")

    def test_prohibited(self):
        configurator = Configurator.from_config_dict(configurations["prohibited"])
        with self.assertRaises(ConfigurationException,
                               msg="Config with prohibited config option does not throw an error!"):
            configurator.get_verified_config()

    def test_required(self):
        configurator = Configurator.from_config_dict(configurations["required"])
        with self.assertRaises(ConfigurationException,
                               msg="Config with missing required config option does not throw an error!"):
            configurator.get_verified_config()

    def test_wrong_model(self):
        config_dict = configurations["minimal"]
        config_dict["model_choice"] = "RNN123"
        configurator = Configurator.from_config_dict(config_dict)

        with self.assertRaises(ConfigurationException,
                               msg="Config with arbitrary model does not throw an exception"):
            configurator.get_verified_config()

        config_dict["model_choice"] = "LightAttention"
        configurator = Configurator.from_config_dict(config_dict)

        with self.assertRaises(ConfigurationException,
                               msg="Config with unavailable model for the protocol does not throw an exception"):
            configurator.get_verified_config()

    def test_non_existing_embedder(self):
        config_dict = configurations["minimal"]
        config_dict["embedder_name"] = "two_hot_encodings"
        configurator = Configurator.from_config_dict(config_dict)

        with self.assertRaises(ConfigurationException,
                               msg="Config with non_existing_embedder does not throw an exception"):
            configurator.get_verified_config()

    def test_download(self):
        config_dict = configurations["download"]
        configurator = Configurator.from_config_dict(config_dict)

        # Download allowed but must not work from arbitrary URL
        with self.assertRaisesRegex(Exception, expected_regex="Could not download",
                                    msg="Config downloading a sequence file does not throw Exception"):
            configurator.get_verified_config()

    def test_download_prohibited(self):
        config_dict = configurations["download_prohibited"]
        configurator = Configurator.from_config_dict(config_dict)

        with self.assertRaises(ConfigurationException,
                               msg="Config downloading an embedding script does not throw an exception"):
            configurator.get_verified_config()

    def test_auto_resume_pretrained_model_mutual_exclusive(self):
        config_dict = configurations["auto_resume_pretrained_model_mutual_exclusive"]
        configurator = Configurator.from_config_dict(config_dict)

        with self.assertRaisesRegex(ConfigurationException, expected_regex="mutual exclusive",
                                    msg="Config with auto_resume and pretrained_model does not throw an error"):
            configurator.get_verified_config()

    def test_embeddings_file_embedder_name_mutual_exclusive(self):
        config_dict = configurations["embeddings_file_embedder_name_mutual_exclusive"]
        configurator = Configurator.from_config_dict(config_dict)

        with self.assertRaisesRegex(ConfigurationException, expected_regex="mutual exclusive",
                                    msg="Config with embeddings file and embedder name does not throw an error"):
            configurator.get_verified_config()

    def test_local_custom_embedder(self):
        configurator = Configurator.from_config_dict(configurations["local_custom_embedder"])
        self.assertTrue(configurator.get_verified_config(), "Local custom embedder config does not work!")

    def test_file_paths_absolute_after_verification(self):
        config_file_paths = configurations["file_paths"]
        configurator = Configurator.from_config_dict(config_file_paths)
        verified_config = configurator.get_verified_config()
        for key in config_file_paths.keys():
            if key != "protocol":
                self.assertTrue(Path(verified_config[key]).is_absolute(),
                                "File paths not absolute after verification!")

    def test_hold_out(self):
        config_dict = {**configurations["minimal"], **configurations["hold_out"]}
        configurator = Configurator.from_config_dict(config_dict)
        self.assertTrue(configurator.get_verified_config(), "Hold out cross validation does not work!")

    def test_k_fold(self):
        config_dict = {**configurations["minimal"], **configurations["k_fold"]}
        configurator = Configurator.from_config_dict(config_dict)
        self.assertTrue(configurator.get_verified_config(), "k_fold does not work!")

        with self.assertRaises(ConfigurationException,
                               msg="Config with missing k_fold params does not throw an exception"):
            config_dict["cross_validation_config"].pop("k")
            configurator.from_config_dict(config_dict).get_verified_config()

    def test_nested_k_fold(self):
        config_dict = {**configurations["multiple_values"], **configurations["nested_k_fold"]}
        original_config = deepcopy(config_dict)

        configurator = Configurator.from_config_dict(config_dict)
        self.assertTrue(configurator.get_verified_config(), "nested k_fold does not work!")

        with self.assertRaises(ConfigurationException,
                               msg="Config with missing nested k_fold param does not throw an exception"):
            config_dict["cross_validation_config"].pop("nested_k")
            configurator.from_config_dict(config_dict).get_verified_config()

        config_dict = deepcopy(original_config)
        with self.assertRaises(ConfigurationException,
                               msg="Config with missing search method for nested k_fold does not throw an exception"):
            config_dict["cross_validation_config"].pop("search_method")
            configurator.from_config_dict(config_dict).get_verified_config()

    def test_leave_p_out(self):
        config_dict = {**configurations["minimal"], **configurations["leave_p_out"]}
        configurator = Configurator.from_config_dict(config_dict)
        self.assertTrue(configurator.get_verified_config(), "leave_p_out does not work!")

        with self.assertRaises(ConfigurationException,
                               msg="Config with missing p param does not throw an exception"):
            config_dict["cross_validation_config"].pop("p")
            configurator.from_config_dict(config_dict).get_verified_config()

    def test_multiple_values(self):
        config_dict = {**configurations["multiple_values"], **configurations["nested_k_fold"]}
        # Test list comprehension
        config_dict["learning_rate"] = "[10**-x for x in [2, 3, 4]]"
        configurator = Configurator.from_config_dict(config_dict)
        self.assertTrue(configurator.get_verified_config(), "List comprehension for hyperparameter optimization"
                                                            "does not work!")
        # Test range expression
        config_dict["batch_size"] = "range(64, 132, 4)"
        configurator = Configurator.from_config_dict(config_dict)
        self.assertTrue(configurator.get_verified_config(), "Range expression for hyperparameter optimization"
                                                            "does not work!")
        # Check that with missing nested k fold cross validation, optimization does not work
        config_dict = {**configurations["multiple_values"], **configurations["hold_out"]}
        with self.assertRaises(ConfigurationException,
                               msg="Config with multiple values for hold_out cv does not "
                                   "throw an exception!"):
            configurator.from_config_dict(config_dict).get_verified_config()

        config_dict = {**configurations["multiple_values"], **configurations["k_fold"]}
        with self.assertRaises(ConfigurationException,
                               msg="Config with multiple values for k_fold (not nested) cv does not "
                                   "throw an exception!"):
            configurator.from_config_dict(config_dict).get_verified_config()

        config_dict = {**configurations["multiple_values"], **configurations["leave_p_out"]}
        with self.assertRaises(ConfigurationException,
                               msg="Config with multiple values for leave_p_out cv does not throw an exception!"):
            configurator.from_config_dict(config_dict).get_verified_config()