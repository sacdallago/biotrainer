import time
import datetime

from pathlib import Path

from ..hp_manager import HyperParameterManager
from ..pipeline import PipelineStep, PipelineContext
from ..pipeline.pipeline_step import PipelineStepType

from ...protocols import Protocol
from ...utilities import seed_all, get_logger, __version__, setup_logging, get_device, calculate_model_hash

logger = get_logger(__name__)


class SetupStep(PipelineStep):

    def get_step_type(self) -> PipelineStepType:
        return PipelineStepType.SETUP

    @staticmethod
    def _post_process_config(context: PipelineContext):
        context.config["protocol"] = Protocol.from_string(context.config["protocol"])

        # Create output dir
        output_dir = Path(context.config["output_dir"])
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
        context.config["output_dir"] = output_dir

        # Setup logging
        setup_logging(str(output_dir), context.config["num_epochs"])

        # Create log directory (if necessary)
        embedder_name = context.config["embedder_name"].split("/")[-1]
        log_dir = output_dir / context.config["model_choice"] / embedder_name
        if not log_dir.is_dir():
            logger.info(f"Creating log-directory: {log_dir}")
            log_dir.mkdir(parents=True)
        context.config["log_dir"] = str(log_dir)

        # Get device once at the beginning
        device = get_device(context.config["device"] if "device" in context.config.keys() else None)
        context.config["device"] = device

        # Set input data
        context.input_data = context.config.get("input_file", None) or context.config.get("input_data", None)
        assert context.input_data is not None, "input_file or input_data must be provided in the config!"

    def process(self, context: PipelineContext) -> PipelineContext:
        context.pipeline_start_time = time.perf_counter()
        pipeline_start_time_abs = str(datetime.datetime.now().isoformat())

        self._post_process_config(context)

        # Log version
        logger.info(f"** Running biotrainer (v{__version__}) training routine **")
        context.output_manager.add_derived_values({'biotrainer_version': str(__version__)})
        # Log start time
        logger.info(f"Pipeline start time: {pipeline_start_time_abs}")
        context.output_manager.add_derived_values({'pipeline_start_time': pipeline_start_time_abs})

        if "pretrained_model" in context.config.keys():
            logger.info(f"Using pre_trained model: {context.config['pretrained_model']}")

        # Create hyperparameter manager
        hp_manager = HyperParameterManager(**context.config)
        context.hp_manager = hp_manager

        # Calculate model hash
        model_hash = calculate_model_hash(dataset_files=[Path(val) for key, val in context.config.items()
                                                         if "_file" in key and Path(str(val)).exists()],
                                          config=context.config,
                                          custom_trainer=context.custom_pipeline
                                          )
        context.model_hash = model_hash
        logger.info(f"Training model with hash: {model_hash}")
        context.output_manager.add_derived_values({"model_hash": model_hash})
        # Seed
        seed = context.config["seed"]
        seed_all(seed)
        logger.info(f"Using seed: {seed}")
        # Log device
        logger.info(f"Using device: {context.config['device']}")

        context.output_manager.add_config(context.config)
        return context