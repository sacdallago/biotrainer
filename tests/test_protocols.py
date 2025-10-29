import os
import tempfile

from ruamel import yaml
from pathlib import Path
from typing import Any, Dict

from biotrainer.utilities.cli import train
from biotrainer.config import ConfigurationException

protocol_to_input = {
    'residue_to_class': {'input_file': "test_input_files/r2c/input.fasta",
                         },
    'residue_to_class-error1': {'input_file': "test_input_files/r2c_error1/sequences.fasta",
                                # Missing target in input file
                                },
    'residue_to_class-error2': {'input_file': "test_input_files/r2c_error2/sequences.fasta",
                                # Sequence and labels length mismatch
                                },
    'residue_to_value': {'input_file': "test_input_files/r2v/input.fasta",
                         },
    'residues_to_class': {'input_file': "test_input_files/s2c/sequences.fasta",
                          },
    'residues_to_value': {'input_file': "test_input_files/s2v/sequences.fasta",
                          },
    'sequence_to_class': {'input_file': "test_input_files/s2c/sequences.fasta",
                          },
    'sequence_to_class-interactionmultiply': {'input_file': "test_input_files/ppi/interactions.fasta",
                                              'interaction': "multiply"},
    'sequence_to_class-interactionconcat': {'input_file': "test_input_files/ppi/interactions.fasta",
                                            'interaction': "concat"},
    'sequence_to_value': {'input_file': "test_input_files/s2v/sequences.fasta", },

}


def setup_config(protocol: str, model_choice: str, embedder_name: str, tmp_config_dir: str) -> Dict[str, Any]:
    template_config_file_path = "test_config.yml"
    with open(template_config_file_path, "r") as config_file:
        config: dict = yaml.load(config_file, Loader=yaml.Loader)
        input_path_absolute = str(Path(protocol_to_input[protocol]["input_file"]).absolute())
        config["input_file"] = input_path_absolute
        actual_protocol = protocol.split("-")[0]
        if len(protocol.split("-")) > 1 and "interaction" in protocol.split("-")[1]:
            config["interaction"] = protocol_to_input[protocol]["interaction"]
        config["protocol"] = actual_protocol
        config["model_choice"] = model_choice
        config["embedder_name"] = embedder_name
        config["output_dir"] = tmp_config_dir
        config["device"] = "cpu"  # Only use cpu for testing
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
            result = train(config=config)
            assert "test_results" in result, "Result does not contain test set metrics!"
            assert os.path.exists(f"{tmp_dir_name}/out.yml"), "No output file generated, run failed!"
        except ConfigurationException:
            assert should_fail, "A ConfigurationException was thrown although it shouldn't have."
        except Exception:
            assert should_fail, "An exception was thrown although it shouldn't have."
