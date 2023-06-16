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


def _download_file(url: str, script_path: str, key_name: str) -> str:
    try:
        logger.info(f"Trying to download {key_name} from {url}")
        req = request.Request(url, headers={
            'User-Agent': "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"
        })

        file_name = url.split("/")[-1]
        save_path = script_path + "/downloaded_" + file_name
        with request.urlopen(req) as response, open(save_path, 'wb') as outfile:
            if response.status == 200:
                logger.info(f"OK - Downloading file {key_name} (size: {response.length / pow(2, 20)} MB)..")
            shutil.copyfileobj(response, outfile)
        logger.info(f"{key_name} successfully downloaded and stored at {save_path}.")
        return save_path
    except Exception as e:
        raise Exception(f"Could not download {key_name} from url {url}") from e


def parse_config_file_and_execute_run(config_file_path: str):
    validate_file(config_file_path)

    # read configuration and execute
    config = read_config_file(config_file_path)

    # Create output directory (if necessary)
    input_file_path = Path(os.path.dirname(os.path.abspath(config_file_path)))
    if "output_dir" in config.keys():
        output_dir = input_file_path / Path(config["output_dir"])
    else:
        output_dir = input_file_path / "output"
    if not output_dir.is_dir():
        logger.info(f"Creating output dir: {output_dir}")
        output_dir.mkdir(parents=True)
    config["output_dir"] = str(output_dir)

    configurator = Configurator.from_config_path(config_file_path)
    configurator.verify_config()

    # Setup logging
    _setup_logging(str(output_dir))

    # Verify config by protocol and check cross validation config
    verify_config(config, __PROTOCOLS)

    # Make input file paths absolute, download files if necessary
    input_file_keys = ["sequence_file", "labels_file", "mask_file", "embeddings_file", "pretrained_model"]
    for input_file_key in input_file_keys:
        if input_file_key in config.keys():
            if urlparse(config[input_file_key]).scheme in ["http", "https", "ftp"]:
                save_path = _download_file(url=config[input_file_key], script_path=str(input_file_path),
                                           key_name=input_file_key)
                config[input_file_key] = save_path
            config[input_file_key] = str(input_file_path / config[input_file_key])

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
