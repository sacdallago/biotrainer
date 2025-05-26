from pathlib import Path
from typing import Union, Dict, Any, Optional, Callable, List

from .cuda_device import get_device
from .model_hash import calculate_model_hash
from .logging import get_logger, setup_logging, clear_logging
from ..config.training_config import training_config

from ..protocols import Protocol
from ..config import Configurator
from ..trainers import Trainer, HyperParameterManager
from ..output_files import OutputManager, output_observer_factory, BiotrainerOutputObserver


def parse_config_file_and_execute_run(config: Union[str, Path, Dict[str, Any]],
                                      custom_trainer_function: Optional[Callable] = None,
                                      custom_output_observers: Optional[List[BiotrainerOutputObserver]] = None) \
        -> Dict[str, Any]:
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
    setup_logging(str(output_dir), config["num_epochs"])
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

    # Create hyperparameter manager
    hp_manager = HyperParameterManager(**config)

    # Calculate model hash
    model_hash = calculate_model_hash(dataset_files=[Path(val) for key, val in config.items()
                                                     if "_file" in key and Path(str(val)).exists()],
                                      config=config,
                                      custom_trainer=True if custom_trainer_function else False
                                      )

    output_observers = output_observer_factory(output_dir=output_dir, config=config)
    if custom_output_observers and len(custom_output_observers) > 0:
        output_observers.extend(custom_output_observers)

    output_manager = OutputManager(observers=output_observers)
    output_manager.add_config(config=config)

    trainer: Trainer
    if custom_trainer_function:
        output_manager.add_derived_values(derived_values={"custom_trainer": True})
        trainer = custom_trainer_function(hp_manager, training_config, model_hash, output_manager, config)
    else:
        # Run biotrainer pipeline
        trainer = Trainer(hp_manager=hp_manager,
                          training_config=config,
                          model_hash=model_hash,
                          output_manager=output_manager,
                          **config
                          )
    output_manager = trainer.training_and_evaluation_routine()

    # Save output_variables in out.yml
    output_result = output_manager.write_to_file(output_dir=output_dir)

    clear_logging()

    return output_result
