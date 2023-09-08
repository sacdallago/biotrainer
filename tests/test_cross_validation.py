import os
import tempfile

from ruamel import yaml
from pathlib import Path
from biotrainer.config import ConfigurationException
from biotrainer.utilities.cli import headless_main as biotrainer_headless_main


def test_cross_validation(cv_config: dict):
    print(f"TESTING CONFIG: {cv_config}")
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_config_path = tmp_dir_name + "/tmp_config_file.yml"
        # Add cv_config to base_config
        with open("test_config.yml", "r") as base_config_file:
            base_config = yaml.safe_load(base_config_file)

        if cv_config["method"] == "k_fold" and cv_config["nested"] is True:
            # Add optimizable hyperparameter for nested k_fold cross validation
            base_config["learning_rate"] = [1e-3, 1e-4]
        else:
            base_config["learning_rate"] = 1e-3
        base_config["cross_validation_config"] = cv_config
        base_config["sequence_file"] = str(Path("test_input_files/cv_s2v/meltome_cv_test.fasta").absolute())
        with open(tmp_config_path, "w") as tmp_config_file:
            tmp_config_file.write(yaml.dump(base_config))

        try:
            biotrainer_headless_main(config_file_path=str(Path(tmp_config_path).absolute()))
            assert os.path.exists(f"{tmp_dir_name}/output/out.yml"), "No output file generated, run failed!"
        except ConfigurationException:
            assert False, "A ConfigurationException was thrown although it shouldn't have."
        except Exception as e:
            assert False, "An exception was thrown although it shouldn't have."
