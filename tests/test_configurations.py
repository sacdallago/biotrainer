import unittest

from pathlib import Path
from copy import deepcopy
from biotrainer.config import Configurator, ConfigurationException

configurations = {
    "minimal": {
        "input_file": "test_input_files/r2c/input.fasta",
        "protocol": "sequence_to_class",
    },
    "required": {
        # Missing input file
        "protocol": "residue_to_class"
    },
    "download": {
        "input_file": "https://example.com/sequences.fasta",
        "protocol": "sequence_to_class",
        "model_choice": "FNN",
    },
    "download_prohibited": {
        "input_file": "test_input_files/r2c/input.fasta",
        "protocol": "sequence_to_class",
        "model_choice": "FNN",
        "embedder_name": "https://example.com/payload.py"
    },
    "auto_resume_pretrained_model_mutual_exclusive": {
        "input_file": "test_input_files/r2c/input.fasta",
        "protocol": "sequence_to_class",
        "auto_resume": True,
        "pretrained_model": "placeholder.pt"
    },
    "mutual_exclusive_device_half_precision": {
        "input_file": "test_input_files/r2c/input.fasta",
        "protocol": "sequence_to_class",
        "device": "cpu",
        "use_half_precision": True
    },
    "local_custom_embedder": {
        "input_file": "test_input_files/r2c/input.fasta",
        "protocol": "sequence_to_class",
        "embedder_name": "../examples/custom_embedder/ankh/ankh_embedder.py"
    },
    "file_paths": {
        "input_file": "test_input_files/r2c/input.fasta",
        "embedder_name": "../examples/custom_embedder/esm2/esm2_embedder.py",
        "output_dir": "test_output",
        "protocol": "residue_to_class"
    },
    "multiple_values": {
        "input_file": "test_input_files/r2c/input.fasta",
        "protocol": "sequence_to_class",
        "model_choice": ["FNN", "DeeperFNN"],
        "optimizer_choice": ["adam", "sgd"],
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
    },
    "hf_valid_for_sequences": {
        "protocol": "sequence_to_class",
        "hf_dataset": {
            "path": "heispv/protein_data_test",
            "subset": "split_1",
            "sequence_column": "protein_sequence",
            "target_column": "protein_class"
        }
    },
    "hf_valid_for_residues": {
        "protocol": "residue_to_class",
        "hf_dataset": {
            "path": "heispv/protein_data_test",
            "subset": "split_3",
            "sequence_column": "protein_sequence",
            "target_column": "secondary_structure"
        }
    },
    "hf_no_subset_required": {
        "protocol": "residue_to_class",
        "hf_dataset": {
            "path": "heispv/protein_data_test_2",
            "subset": "random_subset_name",
            "sequence_column": "protein_sequence",
            "target_column": "secondary_structure"
        }
    },
    "finetuning_ohe": {
        "protocol": "sequence_to_class",
        "embedder_name": "one_hot_encoding",
        "finetuning_config": {
            "method": "lora",
        }
    },
    "finetuning_no_method": {
        "protocol": "sequence_to_class",
        "embedder_name": "one_hot_encoding",
        "finetuning_config": {
            "lora_r": 10,
        }
    }
}


class ConfigurationVerificationTests(unittest.TestCase):

    def test_minimal_configuration(self):
        configurator = Configurator.from_config_dict(configurations["minimal"])
        self.assertTrue(configurator.get_verified_config(), "Minimal config does not work!")

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

    def test_mutual_exclusive_cpu_half_precision(self):
        configurator = Configurator.from_config_dict(configurations["mutual_exclusive_device_half_precision"])
        with self.assertRaises(ConfigurationException,
                               msg="Config with prohibited config combination does not throw an error!"):
            configurator.get_verified_config()

    def test_local_custom_embedder(self):
        configurator = Configurator.from_config_dict(configurations["local_custom_embedder"])
        self.assertTrue(configurator.get_verified_config(), "Local custom embedder config does not work!")

    def test_file_paths_absolute_after_verification(self):
        config_file_paths = configurations["file_paths"]
        configurator = Configurator.from_config_dict(config_file_paths)
        verified_config = configurator.get_verified_config()
        for key in config_file_paths.keys():
            if key != "protocol" and "file" in key:
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
        config_dict = {**configurations["multiple_values"], **configurations["nested_k_fold"],
                       "learning_rate": "[10**-x for x in [2, 3, 4]]"}
        # Test list comprehension
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

    def test_hf_valid_3_split(self):
        config_dict = deepcopy(configurations["hf_valid_for_sequences"])

        configurator = Configurator.from_config_dict(config_dict)
        self.assertTrue(
            configurator.get_verified_config(),
            "Valid hf_dataset configuration for sequences failed."
        )

    def test_hf_invalid_1_split(self):
        config_dict = deepcopy(configurations["hf_valid_for_sequences"])
        config_dict["hf_dataset"]["subset"] = "split_2"

        configurator = Configurator.from_config_dict(config_dict)
        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()

    def test_hf_valid_residues_protocol(self):
        config_dict = deepcopy(configurations["hf_valid_for_residues"])

        configurator = Configurator.from_config_dict(config_dict)
        self.assertTrue(
            configurator.get_verified_config(),
            "Valid hf_dataset configuration for residues failed."
        )

    def test_hf_missing_sequence_column_name(self):
        config_dict = deepcopy(configurations["hf_valid_for_sequences"])
        del config_dict["hf_dataset"]["sequence_column"]

        configurator = Configurator.from_config_dict(config_dict)
        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()

    def test_hf_missing_labels_column_name(self):
        config_dict = deepcopy(configurations["hf_valid_for_sequences"])
        del config_dict["hf_dataset"]["target_column"]

        configurator = Configurator.from_config_dict(config_dict)
        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()

    def test_hf_invalid_sequence_column_name(self):
        config_dict = deepcopy(configurations["hf_valid_for_sequences"])
        config_dict["hf_dataset"]["sequence_column"] = "random_invalid_name"

        configurator = Configurator.from_config_dict(config_dict)
        with self.assertRaises(Exception) as context:
            configurator.get_verified_config()

    def test_hf_invalid_target_column_name(self):
        config_dict = deepcopy(configurations["hf_valid_for_sequences"])
        config_dict["hf_dataset"]["target_column"] = "random_invalid_name"

        configurator = Configurator.from_config_dict(config_dict)
        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()

    def test_hf_invalid_path(self):
        config_dict = deepcopy(configurations["hf_valid_for_sequences"])
        config_dict["hf_dataset"]["path"] = "random_invalid_name"

        configurator = Configurator.from_config_dict(config_dict)
        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()

    def test_hf_invalid_subset(self):
        config_dict = deepcopy(configurations["hf_valid_for_sequences"])
        config_dict["hf_dataset"]["subset"] = "random_invalid_name"

        configurator = Configurator.from_config_dict(config_dict)
        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()

    def test_hf_requires_subset(self):
        config_dict = deepcopy(configurations["hf_valid_for_sequences"])
        del config_dict["hf_dataset"]["subset"]

        configurator = Configurator.from_config_dict(config_dict)
        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()

    def test_hf_not_requires_subset(self):
        config_dict = deepcopy(configurations["hf_no_subset_required"])

        configurator = Configurator.from_config_dict(config_dict)
        print(configurator._config_dict)
        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()


    def test_hf_mutual_exclusive_input_file_name(self):
        config_dict = deepcopy(configurations["hf_valid_for_sequences"])
        config_dict["input_file"] = "random_invalid_name"

        configurator = Configurator.from_config_dict(config_dict)
        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()


    def test_dimension_reduction_methods(self):
        config_dict = deepcopy(configurations["minimal"])
        # Existing method works
        config_dict["dimension_reduction_method"] = "umap"

        configurator = Configurator.from_config_dict(config_dict)

        self.assertTrue(
            configurator.get_verified_config(),
            "Valid dimension_reduction_method: umap failed!"
        )

        # Non-existing method does not work
        config_dict["dimension_reduction_method"] = "nonexistingmethod"
        configurator = Configurator.from_config_dict(config_dict)

        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()

    def test_dimension_reduction_components(self):
        config_dict = deepcopy(configurations["minimal"])
        config_dict["dimension_reduction_method"] = "umap"
        # Positive integers are valid
        valid_values = [22, 1, 44, 123, 99, 1000]
        for valid_value in valid_values:
            config_dict["n_reduced_components"] = valid_value

            configurator = Configurator.from_config_dict(config_dict)
            self.assertTrue(
                configurator.get_verified_config(),
                f"Valid n_reduced_components: {valid_value} failed"
            )

        # Other values are not valid
        invalid_values = [0, -50, 5.5, -10.25, 33.999, 44.000]
        for invalid_value in invalid_values:
            config_dict["n_reduced_components"] = invalid_value
            configurator = Configurator.from_config_dict(config_dict)

            with self.assertRaises(ConfigurationException) as context:
                configurator.get_verified_config()

    def test_bootstrapping_iterations(self):
        config_dict = deepcopy(configurations["minimal"])
        # Positive integers and zero are valid
        valid_values = [0, 1, 44, 123]
        for valid_value in valid_values:
            config_dict["bootstrapping_iterations"] = valid_value

            configurator = Configurator.from_config_dict(config_dict)
            self.assertTrue(
                configurator.get_verified_config(),
                f"Valid bootstrapping_iterations: {valid_value} failed"
            )

        # Other values are not valid
        invalid_values = [-1, 5.5, -2.1, 0.1, 1.00000]
        for invalid_value in invalid_values:
            config_dict["bootstrapping_iterations"] = invalid_value
            configurator = Configurator.from_config_dict(config_dict)

            with self.assertRaises(ConfigurationException) as context:
                configurator.get_verified_config()

    def test_finetuning_ohe(self):
        config_dict = deepcopy(configurations["finetuning_ohe"])
        configurator = Configurator.from_config_dict(config_dict)

        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()

    def test_finetuning_no_method(self):
        config_dict = deepcopy(configurations["finetuning_no_method"])
        configurator = Configurator.from_config_dict(config_dict)

        with self.assertRaises(ConfigurationException) as context:
            configurator.get_verified_config()
