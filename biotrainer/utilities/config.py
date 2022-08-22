import os

from pathlib import Path
from typing import Union

from ruamel import yaml
from ruamel.yaml import YAMLError
from ruamel.yaml.comments import CommentedBase

from ..models import get_all_available_models


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


def verify_config(config: dict, protocols: set):
    protocol = config["protocol"]

    if protocol not in protocols:
        raise ConfigurationException(f"Unknown protocol: {protocol}")

    model = config["model_choice"]

    if "residue_" in protocol:
        required_files = ["labels_file", "sequence_file"]
        for required_file in required_files:
            if required_file not in config.keys():
                raise ConfigurationException(f"Required {required_file} not included in {protocol}")
    elif "sequence_" in protocol:
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

    if model not in get_all_available_models().get(protocol):
        raise ConfigurationException("Model " + model + " not available for protocol: " + protocol)


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
