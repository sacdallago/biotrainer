import os
import logging

from copy import deepcopy
from typing import Dict, Any
from pathlib import Path

from .config import read_config_file, write_config_file
from ..trainers import residue_to_class, sequence_to_class

logger = logging.getLogger(__name__)


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
            raise Exception(f"The file at '{file_path}' is empty")
    except (OSError, TypeError) as e:
        raise Exception(f"The configuration file at '{file_path}' does not exist") from e


__PROTOCOLS = {
    'residue_to_class': residue_to_class,
    'sequence_to_class': sequence_to_class
}


def execute(output_dir: str = "output", protocol: str = "residue_to_class", **kwargs) -> Dict[str, Any]:
    output_vars = deepcopy(locals())
    output_vars.pop('kwargs')

    output_dir = Path(output_dir)

    if not output_dir.is_dir():
        logger.info(f"Creating output dir: {output_dir}")
        output_dir.mkdir(parents=True)

    return __PROTOCOLS.get(protocol)(**{**kwargs, **output_vars})


def parse_config_file_and_execute_run(config_file_path: str):
    _validate_file(config_file_path)

    # read configuration and execute
    config = read_config_file(config_file_path)
    input_file_path = Path(os.path.dirname(os.path.abspath(config_file_path)))
    config["labels_file"] = input_file_path / config["labels_file"]
    config["sequence_file"] = input_file_path / config["sequence_file"]

    original_config = deepcopy(config)
    out_config = execute(output_dir=str(input_file_path / "output"), **original_config)
    write_config_file(
        str(Path(out_config['output_dir']) / "out.yml"),
        out_config
    )
