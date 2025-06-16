from pathlib import Path
from typing import Union, Dict, Any, Optional, List

from .logging import clear_logging

from ..config import Configurator
from ..trainers import Trainer, Pipeline
from ..output_files import OutputManager, output_observer_factory, BiotrainerOutputObserver


def parse_config_file_and_execute_run(config: Union[str, Path, Dict[str, Any]],
                                      custom_pipeline: Optional[Pipeline] = None,
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
    output_dir = Path(config["output_dir"])

    output_observers = output_observer_factory(output_dir=output_dir, config=config)
    if custom_output_observers and len(custom_output_observers) > 0:
        output_observers.extend(custom_output_observers)

    output_manager = OutputManager(observers=output_observers)
    output_manager.add_config(config=config)

    trainer: Trainer
    if custom_pipeline:
        output_manager.add_derived_values(derived_values={"custom_pipeline": True})

    # Run biotrainer pipeline
    trainer = Trainer(config=config,
                      output_manager=output_manager,
                      custom_pipeline=custom_pipeline
                      )

    output_manager = trainer.run()

    # Save output_variables in out.yml
    output_result = output_manager.write_to_file(output_dir=output_dir)

    clear_logging()

    return output_result
