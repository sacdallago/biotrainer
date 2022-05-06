import os
import logging

from copy import deepcopy
from typing import Dict, Any
from pathlib import Path

from ..models import __MODELS
from ..trainers import get_trainer
from .config import read_config_file, write_config_file

logger = logging.getLogger(__name__)


class ConfigurationException(Exception):
    pass


def _validate_file(file_path: str):
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


def _verify_config(config: dict):
    protocol = config["protocol"]
    model = config["model_choice"]

    if protocol == 'residue_to_class':
        try:
            labels_file = config["labels_file"]
            sequence_file = config["sequence_file"]
        except KeyError as e:
            raise ConfigurationException("Correct files not available for protocol: " + protocol)

    elif protocol == 'sequence_to_class':
        try:
            sequence_file = config["sequence_file"]
        except KeyError as e:
            raise ConfigurationException("Correct files not available for protocol: " + protocol)

        if "labels_file" in config.keys() and config["labels_file"] != "":
            raise ConfigurationException("Labels are expected to be found in the sequence file for protocol: "
                                         + protocol)

    if model not in __MODELS.get(protocol):
        raise ConfigurationException( "Model " + model + " not available for protocol: " + protocol)


def _convert_paths_to_absolute(config: dict, input_file_path: Path):
    if "labels_file" in config.keys():
        config["labels_file"] = input_file_path / config["labels_file"]
    if "sequence_file" in config.keys():
        config["sequence_file"] = input_file_path / config["sequence_file"]
    return config


__PROTOCOLS = {
    'residue_to_class',
    'sequence_to_class'
}


def execute(output_dir: str = "output", protocol: str = "residue_to_class", **kwargs) -> Dict[str, Any]:
    output_vars = deepcopy(locals())
    output_vars.pop('kwargs')

    output_dir = Path(output_dir)

    if not output_dir.is_dir():
        logger.info(f"Creating output dir: {output_dir}")
        output_dir.mkdir(parents=True)

    return get_trainer(**{**kwargs, **output_vars})


def parse_config_file_and_execute_run(config_file_path: str):
    _validate_file(config_file_path)

    # read configuration and execute
    config = read_config_file(config_file_path)
    _verify_config(config)

    input_file_path = Path(os.path.dirname(os.path.abspath(config_file_path)))
    config = _convert_paths_to_absolute(config, input_file_path)

    original_config = deepcopy(config)
    out_config = execute(output_dir=str(input_file_path / "output"), **original_config)
    write_config_file(
        str(Path(out_config['output_dir']) / "out.yml"),
        out_config
    )
