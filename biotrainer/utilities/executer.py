import os
import shutil
import logging
import torch._dynamo as dynamo

from pathlib import Path
from copy import deepcopy
from urllib import request
from urllib.parse import urlparse

from .cuda_device import get_device
from .config import validate_file, read_config_file, verify_config, write_config_file, add_default_values_to_config
from ..config import Configurator
from ..protocols import Protocol

from ..trainers import Trainer, HyperParameterManager

logger = logging.getLogger(__name__)

__PROTOCOLS = {
    'residue_to_class',
    'residues_to_class',
    'sequence_to_class',
    'sequence_to_value',
}


def _setup_logging(output_dir: str):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[
                            logging.FileHandler(output_dir + "/logger_out.log"),
                            logging.StreamHandler()]
                        )
    logging.captureWarnings(True)
    # Suppress info logging from TorchDynamo (torch.compile(model)) with logging.ERROR
    dynamo.config.log_level = logging.INFO


def parse_config_file_and_execute_run(config_file_path: str):
    configurator = Configurator.from_config_path(config_file_path)
    # Setup logging
    output_dir = configurator.get_output_dir()
    _setup_logging(str(output_dir))

    config = configurator.get_verified_config()
    config["protocol"] = Protocol[config["protocol"]]

    if "pretrained_model" in config.keys():
        logger.info(f"Using pre_trained model: {config['pretrained_model']}")

    # Add default hyper_parameters to config if not defined by user
    config = add_default_values_to_config(config, output_dir=str(output_dir))

    # Custom embedder name
    embedder_name = config["embedder_name"]
    if ".py" in embedder_name:
        config["embedder_name"] = str(input_file_path / config["embedder_name"])

    # Create log directory (if necessary)
    log_dir = output_dir / config["model_choice"] / str(embedder_name).replace(".py", "/")
    if not log_dir.is_dir():
        logger.info(f"Creating log-directory: {log_dir}")
        log_dir.mkdir(parents=True)
    config["log_dir"] = str(log_dir)

    # Get device once at the beginning
    device = get_device(config["device"] if "device" in config.keys() else None)
    config["device"] = device

    # Create hyper parameter manager
    hp_manager = HyperParameterManager(**config)

    # Copy output_vars from config
    output_vars = deepcopy(config)

    # Run biotrainer pipeline
    trainer = Trainer(hp_manager=hp_manager,
                      output_vars=output_vars,
                      **config
                      )
    out_config = trainer.training_and_evaluation_routine()

    # Save output_variables in out.yml
    write_config_file(
        str(Path(out_config['output_dir']) / "out.yml"),
        out_config
    )
