import os
import yaml
import shutil
import tempfile

from biotrainer.utilities.config import ConfigurationException
from biotrainer.utilities.cli import headless_main as biotrainer_headless_main

protocol_to_input_files = {
    "residue_to_class": {"sequence_file": "r2c/sequences.fasta",
                         "labels_file": "r2c/labels.fasta"},
    "sequence_to_class": {"sequence_file": "s2c/sequences.fasta",
                          "labels_file": ""}
}


def create_temporary_copy(path):
    tmp = tempfile.NamedTemporaryFile(dir=".", delete=False)
    shutil.copy2(path, tmp.name)
    return tmp.name


def setup_config(protocol: str, model_choice: str, embedder_name: str) -> str:
    config_file_path = "test_config.yml"
    config_file_path = create_temporary_copy(config_file_path)
    with open(config_file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
        config["sequence_file"] = protocol_to_input_files[protocol]["sequence_file"]
        config["labels_file"] = protocol_to_input_files[protocol]["labels_file"]
        config["protocol"] = protocol
        config["model_choice"] = model_choice
        config["embedder_name"] = embedder_name
    with open(config_file_path, "w") as config_file:
        yaml.safe_dump(config, config_file, default_flow_style=False, sort_keys=False)
    return config_file_path


def test_config(protocol: str, model: str, embedder_name: str, should_fail: bool):
    # Clean up before test
    if os.path.exists("output/"):
        shutil.rmtree("output/")

    print("TESTING CONFIG: " + protocol + " - " + model +
          " - " + embedder_name + " - Passed when failed: " + str(should_fail))
    temp_config_file_path = setup_config(protocol=protocol,
                                         model_choice=model,
                                         embedder_name=embedder_name)
    try:
        biotrainer_headless_main(config_file_path=os.path.abspath(temp_config_file_path))
        assert os.path.exists("output/out.yml"), "No output file generated, run failed!"
        # Clean up after test
        shutil.rmtree("output/")
    except ConfigurationException:
        assert should_fail, "A ConfigurationException was thrown although it shouldn't have."
    os.remove(temp_config_file_path)
