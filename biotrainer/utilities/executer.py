import os
import logging

from pathlib import Path
from urllib.parse import urlparse

from .cuda_device import get_device
from .config import validate_file, read_config_file, verify_config, write_config_file, add_default_values_to_config

from ..trainers import download_embeddings, ModelFactory, Trainer

logger = logging.getLogger(__name__)

__PROTOCOLS = {
    'residue_to_class',
    'residues_to_class',
    'sequence_to_class',
    'sequence_to_value',
    'protein_protein_interaction'
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
        embeddings_file = config["embeddings_file"]
        if urlparse(embeddings_file).scheme in ["http", "https", "ftp"]:
            embeddings_file = download_embeddings(url=config["embeddings_file"], script_path=str(input_file_path))

        config["embeddings_file"] = str(input_file_path / embeddings_file)
        config["embedder_name"] = "custom_embeddings"
    if "pretrained_model" in config.keys():
        config["pretrained_model"] = str(input_file_path / config["pretrained_model"])

    # Create output directory (if necessary)
    output_dir = input_file_path / "output"
    if not output_dir.is_dir():
        logger.info(f"Creating output dir: {output_dir}")
        output_dir.mkdir(parents=True)

    # Create log directory (if necessary)
    log_dir = output_dir / config["model_choice"] / config["embedder_name"]
    if not log_dir.is_dir():
        logger.info(f"Creating log-directory: {log_dir}")
        log_dir.mkdir(parents=True)

    # Add default hyper_parameters to config if not defined by user
    config = add_default_values_to_config(config, output_dir=str(output_dir), log_dir=str(log_dir))

    # Find pre-trained model in auto_resume mode
    if config["auto_resume"]:
        if "checkpoint.pt" in os.listdir(log_dir):
            config["pretrained_model"] = str(log_dir / "checkpoint.pt")
        else:
            logger.info("auto_resume is enabled in the configuration file, but no valid checkpoint was found. "
                        "Training new model from scratch.")
    if "pretrained_model" in config.keys():
        logger.info(f"Using pre_trained model: {config['pretrained_model']}")

    # Get device once at the beginning
    device = get_device(config["device"] if "device" in config.keys() else None)
    config["device"] = device

    # Create model factory
    model_factory = ModelFactory(**config)

    # Run biotrainer pipeline
    trainer = Trainer(output_dir=str(output_dir),
                      log_dir=str(log_dir),
                      model_factory=model_factory,
                      **config)
    out_config = trainer.training_and_evaluation_routine()

    # Save output_variables in out.yml
    write_config_file(
        str(Path(out_config['output_dir']) / "out.yml"),
        out_config
    )
