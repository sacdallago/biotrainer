import torch
import logging

from ruamel import yaml
from pathlib import Path
from copy import deepcopy
from typing import Union, Dict, Any
from ruamel.yaml.comments import CommentedBase

from .logging import get_logger
from .cuda_device import get_device

from ..config import Configurator
from ..protocols import Protocol
from ..trainers import Trainer, HyperParameterManager

def _setup_logging(output_dir: str):
    logging.captureWarnings(True)
    biotrainer_logger = logging.getLogger('biotrainer')
    biotrainer_logger.propagate = False  # Prevent propagation to root logger

    # Set up handlers
    file_handler = logging.FileHandler(output_dir + "/logger_out.log")
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    biotrainer_logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

    biotrainer_logger.addHandler(file_handler)
    biotrainer_logger.addHandler(stream_handler)

    # Only log errors for onnx and dynamo
    torch._logging.set_logs(dynamo=logging.ERROR, onnx=logging.ERROR, onnx_diagnostics=False)
    for logger_name in ["torch.onnx", "torch._dynamo", "onnxscript"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


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


def parse_config_file_and_execute_run(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    # Verify config via configurator
    configurator = None
    if isinstance(config, str):
        configurator = Configurator.from_config_path(config)
    elif isinstance(config, Path):
        configurator = Configurator.from_config_path(str(config))
    elif isinstance(config, dict):
        configurator = Configurator.from_config_dict(config)

    assert configurator is not None, f"Config could not be read, incorrect type: {type(config)}"

    config = configurator.get_verified_config(ignore_file_checks=False)
    config["protocol"] = Protocol[config["protocol"]]

    # Create output dir and setup logging
    output_dir = Path(config["output_dir"])
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    _setup_logging(str(output_dir))
    logger = get_logger(__name__)

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
    output_result = trainer.training_and_evaluation_routine()

    # Save output_variables in out.yml
    _write_output_file(
        str(Path(output_result['output_dir']) / "out.yml"),
        output_result
    )
    return output_result
