import os
import logging

from pathlib import Path
from copy import deepcopy
from typing import Dict, Any

from ..trainers import get_trainer
from .config import validate_file, read_config_file, verify_config, write_config_file

logger = logging.getLogger(__name__)


__PROTOCOLS = {
    'residue_to_class',
    'sequence_to_class',
    'sequence_to_value'
}


def execute(output_dir: Path = "output", protocol: str = "residue_to_class", **kwargs) -> Dict[str, Any]:
    output_vars = deepcopy(locals())
    output_vars.pop('kwargs')

    if not output_dir.is_dir():
        logger.info(f"Creating output dir: {output_dir}")
        output_dir.mkdir(parents=True)

    return get_trainer(**{**kwargs, **output_vars})


def parse_config_file_and_execute_run(config_file_path: str):
    validate_file(config_file_path)

    # read configuration and execute
    config = read_config_file(config_file_path)
    verify_config(config, __PROTOCOLS)

    input_file_path = Path(os.path.dirname(os.path.abspath(config_file_path)))
    if "labels_file" in config.keys():
        config["labels_file"] = input_file_path / config["labels_file"]
    if "sequence_file" in config.keys():
        config["sequence_file"] = input_file_path / config["sequence_file"]

    original_config = deepcopy(config)
    out_config = execute(output_dir=input_file_path / "output", **original_config)
    write_config_file(
        str(Path(out_config['output_dir']) / "out.yml"),
        out_config
    )
