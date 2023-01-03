import os
import logging

from pathlib import Path
from typing import Union

from ruamel import yaml
from ruamel.yaml import YAMLError
from ruamel.yaml.comments import CommentedBase

from ..models import get_all_available_models

logger = logging.getLogger(__name__)


class ConfigurationException(Exception):
    """
    Exception for invalid configurations
    """


def parse_config(config_str: str, preserve_order: bool = True) -> dict:
    """
    Parse a configuration string

    Parameters
    ----------
    config_str : str
        Configuration to be parsed
    preserve_order : bool, optional (default: False)
        Preserve formatting of input configuration
        string

    Returns
    -------
    dict
        Configuration dictionary
    """
    try:
        if preserve_order:
            return yaml.load(config_str, Loader=yaml.RoundTripLoader)
        else:
            return yaml.safe_load(config_str)
    except YAMLError as e:
        raise ConfigurationException(
            f"Could not parse configuration file at {config_str} as yaml. "
            "Formatting mistake in config file? "
            "See Error above for details."
        ) from e


def read_config_file(config_path: Union[str, Path], preserve_order: bool = True) -> dict:
    """
    Read config from path to file.

    :param config_path: path to .yml config file
    :param preserve_order:
    :return:
    """
    with open(config_path, "r") as fp:
        try:
            if preserve_order:
                return yaml.load(fp, Loader=yaml.Loader)
            else:
                return yaml.safe_load(fp)
        except YAMLError as e:
            raise ConfigurationException(
                f"Could not parse configuration file at '{config_path}' as yaml. "
                "Formatting mistake in config file? "
                "See Error above for details."
            ) from e


def validate_file(file_path: str):
    """
    Verify if a file exists and is not empty.
    Parameters
    ----------
    file_path : str
        Path to file to check
    Returns
    -------
    bool
        True if file exists and is non-zero size,
        False otherwise.
    """
    try:
        if os.stat(file_path).st_size == 0:
            raise ConfigurationException(f"The file at '{file_path}' is empty")
    except (OSError, TypeError) as e:
        raise ConfigurationException(f"The configuration file at '{file_path}' does not exist") from e


def add_default_values_to_config(config: dict, output_dir: str, log_dir: str):
    default_values = {
        "auto_resume": False,
        "num_epochs": 200,
        "use_class_weights": False,
        "batch_size": 128,
        "embedder_name": 'custom_embeddings',
        "shuffle": True,
        "seed": 42,
        "patience": 10,
        "epsilon": 0.001,
        "learning_rate": 1e-3,
        "output_dir": output_dir,
        "log_dir": log_dir,
        "cross_validation_config": {"method": "hold_out"}
    }
    for default_key, default_value in default_values.items():
        if default_key not in config.keys():
            config[default_key] = default_value

    return config


def verify_config(config: dict, protocols: set):
    # Check protocol and associated files
    protocol = config["protocol"]
    if protocol not in protocols:
        raise ConfigurationException(f"Unknown protocol: {protocol}")

    if "residue_" in protocol:
        required_files = ["labels_file", "sequence_file"]
        for required_file in required_files:
            if required_file not in config.keys():
                raise ConfigurationException(f"Required {required_file} not included in {protocol}")
    elif "sequence_" in protocol or "residues_" in protocol:
        required_files = ["sequence_file"]
        for required_file in required_files:
            if required_file not in config.keys():
                raise ConfigurationException(f"Required {required_file} not included in {protocol}")

        if "labels_file" in config.keys() and config["labels_file"] != "":
            raise ConfigurationException(
                f"Labels are expected to be found in the sequence file for protocol: {protocol}")
        if "mask_file" in config.keys() and config["mask_file"] != "":
            raise ConfigurationException(
                f"Mask file cannot be applied for protocol: {protocol}")

    if "_value" in protocol:
        if "use_class_weights" in config.keys():
            use_class_weights = config["use_class_weights"]
            if type(use_class_weights) == list or eval(str(use_class_weights).capitalize()):
                raise ConfigurationException(f"Class weights are not supported for protocol {protocol}!")

    # Check model for protocol
    model = config["model_choice"]
    if model not in get_all_available_models().get(protocol):
        raise ConfigurationException("Model " + model + " not available for protocol: " + protocol)

    # Check pretraining
    if "auto_resume" in config.keys() and "pretrained_model" in config.keys():
        if config["auto_resume"]:
            raise ConfigurationException("auto_resume and pretrained_model are mutually exclusive.\n"
                                         "Use auto_resume in case you need to "
                                         "restart your training job multiple times.\n"
                                         "Use pretrained_model if you want to continue to train a specific model.")

    # Check embeddings input
    if "embedder_name" in config.keys() and "embeddings_file" in config.keys():
        raise ConfigurationException("embedder_name and embeddings_file are mutually exclusive. "
                                     "Please provide either an embedder_name to calculate embeddings from scratch or "
                                     "an embeddings_file to use pre-computed embeddings.")

    # Check cross validation configuration
    if "cross_validation_config" in config.keys():
        cross_validation_config = config["cross_validation_config"]
        if type(cross_validation_config) is not dict:
            raise ConfigurationException("cross_validation_config must be given as a dict of values, e.g.: \n"
                                         "cross_validation_config:\n"
                                         "\tmethod: k_fold\n"
                                         "\tk: 3\n"
                                         "\tstratified: False\n"
                                         "\tnested: True")
        if "method" not in cross_validation_config.keys():
            raise ConfigurationException(f"No method for cross validation configured!")
        else:
            method = cross_validation_config["method"]
            supported_methods = ["hold_out", "k_fold", "leave_p_out"]
            if method not in supported_methods:
                raise ConfigurationException(f"Unknown method {method} for cross_validation!\n"
                                             f"Supported methods: {supported_methods}")
            if method == "hold_out" and len(cross_validation_config.keys()) > 1:
                raise ConfigurationException(f"cross_validation method hold_out does not support "
                                             f"any additional parameters!\n"
                                             f"Given: {cross_validation_config.keys()}")
            if method != "hold_out" and "pretrained_model" in config.keys():
                raise ConfigurationException(f"pretrained_model option only available for hold_out "
                                             f"cross validation!")
            if method == "k_fold":
                if "k" not in cross_validation_config.keys():
                    raise ConfigurationException(f"Missing parameter k for k-fold cross validation!")
                elif int(cross_validation_config["k"]) < 2:
                    raise ConfigurationException(f"k for k-fold cross_validation must be >= 2!")
                if "nested" in cross_validation_config.keys():
                    nested = eval(str(cross_validation_config["nested"]).capitalize())
                    if nested:
                        if "nested_k" not in cross_validation_config.keys():
                            raise ConfigurationException(
                                f"Missing parameter nested_k for nested k-fold cross validation!")
                        elif int(cross_validation_config["nested_k"]) < 2:
                            raise ConfigurationException(f"nested k for nested k-fold cross_validation must be >= 2!")
                        if "search_method" not in cross_validation_config.keys():
                            raise ConfigurationException(
                                f"Missing parameter search_method for nested k-fold cross validation!"
                            )
                        elif cross_validation_config["search_method"] == "random_search":
                            if "n_max_evaluations_random" not in cross_validation_config.keys():
                                raise ConfigurationException(f"hyper parameter search method random_search misses "
                                                             f"parameter n_max_evaluations_random!")
                            elif int(cross_validation_config["n_max_evaluations_random"] < 2):
                                raise ConfigurationException(
                                    f"n_max_evaluations_random for random_search must be >= 2!")
                        if "repeat" in cross_validation_config.keys() and int(cross_validation_config["repeat"]) > 1:
                            raise ConfigurationException(f"A combination of repeat and nested is currently not "
                                                         f"supported for cross validation!")

                    else:
                        if "nested_k" in cross_validation_config.keys():
                            logger.warning(f"nested == False for k_fold cross validation, but nested_k is given. "
                                           f"nested_k ({cross_validation_config['nested_k']}) is ignored!")
                        if "search_method" in cross_validation_config.keys():
                            logger.warning(f"nested == False for k_fold cross validation, but search_method is given. "
                                           f"search_method ({cross_validation_config['search_method']}) is ignored!")

            if method == "leave_p_out":
                if "p" not in cross_validation_config.keys():
                    raise ConfigurationException(f"Missing parameter p for leave_p_out cross validation!")
                elif int(cross_validation_config["p"]) < 1:
                    raise ConfigurationException(f"p for leave_p_out cross_validation must be >= 1!")
    return True


def write_config_file(out_filename: str, config: dict) -> None:
    """
    Save configuration data structure in YAML file.

    Parameters
    ----------
    out_filename : str
        Filename of output file
    config : dict
        Config data that will be written to file
    """
    if isinstance(config, CommentedBase):
        dumper = yaml.RoundTripDumper
    else:
        dumper = yaml.Dumper

    with open(out_filename, "w") as f:
        f.write(
            yaml.dump(config, Dumper=dumper, default_flow_style=False)
        )
