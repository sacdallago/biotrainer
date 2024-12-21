import os
import tempfile

from ruamel import yaml
from pathlib import Path
from typing import Any, Dict

from biotrainer.config import ConfigurationException
from biotrainer.utilities.cli import headless_main as biotrainer_headless_main

protocol_to_input = {
    'residue_to_class': {'sequence_file': "test_input_files/r2c/sequences.fasta",
                         'labels_file': "test_input_files/r2c/labels.fasta",
                         'loss_choice': "cross_entropy_loss"},
    'residue_to_class-error1': {'sequence_file': "test_input_files/r2c_error1/sequences.fasta",
                                # Missing sequence in labels file
                                'labels_file': "test_input_files/r2c_error1/labels.fasta",
                                'loss_choice': "cross_entropy_loss"},
    'residue_to_class-error2': {'sequence_file': "test_input_files/r2c_error2/sequences.fasta",
                                # Sequence and labels length mismatch
                                'labels_file': "test_input_files/r2c_error2/labels.fasta",
                                'loss_choice': "cross_entropy_loss"},
    'residues_to_class': {'sequence_file': "test_input_files/s2c/sequences.fasta",
                          'loss_choice': "cross_entropy_loss"},
    'residues_to_value': {'sequence_file': "test_input_files/s2v/sequences.fasta",
                          'loss_choice': "mean_squared_error"},
    'sequence_to_class': {'sequence_file': "test_input_files/s2c/sequences.fasta",
                          'loss_choice': "cross_entropy_loss"},
    'sequence_to_class-interactionmultiply': {'sequence_file': "test_input_files/ppi/interactions.fasta",
                                              'loss_choice': "cross_entropy_loss",
                                              'interaction': "multiply"},
    'sequence_to_class-interactionconcat': {'sequence_file': "test_input_files/ppi/interactions.fasta",
                                            'loss_choice': "cross_entropy_loss",
                                            'interaction': "concat"},
    'sequence_to_value': {'sequence_file': "test_input_files/s2v/sequences.fasta",
                          'loss_choice': "mean_squared_error"},

}


def setup_config(protocol: str, model_choice: str, embedder_name: str, tmp_config_dir: str) -> Dict[str, Any]:
    template_config_file_path = "test_config.yml"
    with open(template_config_file_path, "r") as config_file:
        config: dict = yaml.load(config_file, Loader=yaml.Loader)
        sequence_path_absolute = str(Path(protocol_to_input[protocol]["sequence_file"]).absolute())
        config["sequence_file"] = sequence_path_absolute
        if "labels_file" in protocol_to_input[protocol]:
            labels_path_absolute = str(Path(protocol_to_input[protocol]["labels_file"]).absolute())
            config["labels_file"] = labels_path_absolute
        else:
            config.pop("labels_file")
        config["loss_choice"] = protocol_to_input[protocol]["loss_choice"]
        actual_protocol = protocol.split("-")[0]
        if len(protocol.split("-")) > 1 and "interaction" in protocol.split("-")[1]:
            config["interaction"] = protocol_to_input[protocol]["interaction"]
        config["protocol"] = actual_protocol
        config["model_choice"] = model_choice
        config["embedder_name"] = embedder_name
        config["output_dir"] = tmp_config_dir
        return config


def test_protocol_config(protocol: str, model: str, embedder_name: str, should_fail: bool):
    print("TESTING CONFIG: " + protocol + " - " + model +
          " - " + embedder_name + " - Passed when failed: " + str(should_fail))
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        config = setup_config(protocol=protocol,
                              model_choice=model,
                              embedder_name=embedder_name,
                              tmp_config_dir=tmp_dir_name)
        try:
            result = biotrainer_headless_main(config)
            assert "test_iterations_results" in result, "Result does not contain test set metrics!"
            assert os.path.exists(f"{tmp_dir_name}/out.yml"), "No output file generated, run failed!"
        except ConfigurationException:
            assert should_fail, "A ConfigurationException was thrown although it shouldn't have."
        except Exception as e:
            assert should_fail, "An exception was thrown although it shouldn't have."
