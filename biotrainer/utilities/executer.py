import os
import logging

from pathlib import Path

from ..trainers import training_and_evaluation_routine
from .config import validate_file, read_config_file, verify_config, write_config_file

logger = logging.getLogger(__name__)


__PROTOCOLS = {
    'residue_to_class',
    'sequence_to_class',
    'sequence_to_value'
}


def parse_config_file_and_execute_run(config_file_path: str):
    validate_file(config_file_path)

    # read configuration and execute
    config = read_config_file(config_file_path)
    verify_config(config, __PROTOCOLS)

    # Make input file paths absolute
    input_file_path = Path(os.path.dirname(os.path.abspath(config_file_path)))
    if "labels_file" in config.keys():
        config["labels_file"] = str(input_file_path / config["labels_file"])
    if "sequence_file" in config.keys():
        config["sequence_file"] = str(input_file_path / config["sequence_file"])
    if "mask_file" in config.keys():
        config["mask_file"] = str(input_file_path / config["mask_file"])
    if "embeddings_file" in config.keys():
        config["embeddings_file"] = str(input_file_path / config["embeddings_file"])

    # Create output directory (if necessary)
    output_dir = input_file_path / "output"
    if not output_dir.is_dir():
        logger.info(f"Creating output dir: {output_dir}")
        output_dir.mkdir(parents=True)

    # Run biotrainer pipeline
    out_config = training_and_evaluation_routine(output_dir=str(output_dir), **config)

    # Save output_variables in out.yml
    write_config_file(
        str(Path(out_config['output_dir']) / "out.yml"),
        out_config
    )
