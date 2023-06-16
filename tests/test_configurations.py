import unittest

from ruamel import yaml
from biotrainer.utilities import config
from biotrainer.utilities.executer import __PROTOCOLS as PROTOCOLS
from biotrainer.config import Configurator, ConfigurationException

configurations = {
    "prohibited": {
        "sequence_file": "test_input_files/r2c/sequences.fasta",
        "labels_file": "test_input_files/r2c/labels.fasta",
        "protocol": "sequence_to_class"
    },
    "minimal": {
        "sequence_file": "test_input_files/r2c/sequences.fasta",
        "protocol": "sequence_to_class",
        "model_choice": "FNN"
    },
    "k_fold": {
        "cross_validation_config": {
            "method": "k_fold",
            "k": 3,
            "stratified": True,
            "repeated": True
        }
    },
    "nested_k_fold": {
        "cross_validation_config": {
            "method": "k_fold",
            "k": 4,
            "nested": True,
            "nested_k": 3,
            "search_method": "random_search",
            "n_max_evaluations_random": 5
        }
    }
}


class ConfigurationVerificationTests(unittest.TestCase):

    def test_minimal_configuration(self):
        configurator = Configurator.from_config_dict(configurations["minimal"])
        self.assertTrue(configurator.verify_config(), "Minimal config does not work!")

    def test_prohibited(self):
        configurator = Configurator.from_config_dict(configurations["prohibited"])
        with self.assertRaises(ConfigurationException,
                               msg="Config with prohibited config option does not throw an error!"):
            configurator.verify_config()

    def test_wrong_model(self):
        config_dict = configurations["minimal"]
        config_dict["model_choice"] = "RNN123"
        configurator = Configurator.from_config_dict(config_dict)

        with self.assertRaises(ConfigurationException,
                               msg="Config with wrong model does not throw an exception"):
            configurator.verify_config()

    def test_k_fold(self):
        config_dict = config.parse_config(yaml.dump({**configurations["minimal"], **configurations["k_fold"]}))
        config_dict["cross_validation_config"] = dict(config_dict["cross_validation_config"])
        self.assertTrue(config.verify_config(config_dict, PROTOCOLS), "k_fold does not work!")
        with self.assertRaises(config.ConfigurationException,
                               msg="Config with missing k_fold params does not throw an exception"):
            config_dict["cross_validation_config"].pop("k")
            config.verify_config(config_dict, PROTOCOLS)

    def test_nested_k_fold(self):
        config_dict = config.parse_config(yaml.dump({**configurations["minimal"], **configurations["nested_k_fold"]}))
        config_dict["cross_validation_config"] = dict(config_dict["cross_validation_config"])
        self.assertTrue(config.verify_config(config_dict, PROTOCOLS), "nested k_fold does not work!")
        with self.assertRaises(config.ConfigurationException,
                               msg="Config with missing nested k_fold params does not throw an exception"):
            config_dict["cross_validation_config"].pop("nested_k")
            config.verify_config(config_dict, PROTOCOLS)
