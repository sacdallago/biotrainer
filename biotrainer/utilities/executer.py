import logging
import torch._dynamo as dynamo

from ruamel import yaml
from pathlib import Path
from copy import deepcopy
from ruamel.yaml.comments import CommentedBase

from .cuda_device import get_device

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


def _write_output_file(out_filename: str, config: dict) -> None:
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


def parse_config_file_and_execute_run(config_file_path: str):
    # Verify config via configurator
    configurator = Configurator.from_config_path(config_file_path)
    config = configurator.get_verified_config()
    config["protocol"] = Protocol[config["protocol"]]

    # Create output dir and setup logging
    output_dir = Path(config["output_dir"])
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    _setup_logging(str(output_dir))

    if "pretrained_model" in config.keys():
        logger.info(f"Using pre_trained model: {config['pretrained_model']}")

    # Create log directory (if necessary)
    embedder_name = config["embedder_name"].split("/")[-1].replace(".py", "")  # Accounting for custom embedder script
    log_dir = output_dir / config["model_choice"] / embedder_name
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
    _write_output_file(
        str(Path(out_config['output_dir']) / "out.yml"),
        out_config
    )
