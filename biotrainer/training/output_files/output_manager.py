import os
import torch

from ruamel import yaml
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, List, Optional
from biotrainer_core.data_classes import Protocol, EpochMetrics, BiotrainerModelResult, TrainingResult, DerivedValues, \
    TestResult, BiotrainerPrediction, BiotrainerModelUpdate

from .biotrainer_output_observer import BiotrainerOutputObserver

from ..utilities import FeatureScaler

from ...shared import get_logger, get_device, __version__

logger = get_logger(__name__)


class OutputManager:
    """Manages training outputs with type-safe model and observer notifications."""

    def __init__(self, observers: List[BiotrainerOutputObserver]):
        self._observers = observers
        self._model_result = BiotrainerModelResult(
            derived_values=DerivedValues(),
        )

    def _notify_observers(self, data: BiotrainerModelUpdate) -> None:
        for observer in self._observers:
            try:
                observer.update(data)
            except Exception as e:
                logger.error(f"Error in observer: {str(e)}")

    # ============= Config =============
    @staticmethod
    def _convert_config_value(key: str, value: Any) -> Any:
        if key == "input_data":
            return len(value)
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, Protocol):
            return value.name
        if isinstance(value, Path):
            return str(value)
        return value

    def add_config(self, config: Dict[str, Any]) -> None:
        assert len(self._model_result.config) == 0, "Config already set!"
        converted_config = {
            str(k): self._convert_config_value(k, v)
            for k, v in config.items()
        }
        self._model_result.config = converted_config
        self._notify_observers(BiotrainerModelUpdate(current_model_result=self._model_result))

    # ============= Derived Values =============
    def update_derived_values(self, **kwargs) -> None:
        """Update derived values with type checking.

        Example:
            output_manager.update_derived_values(
                class_int2str=target_manager.class_int2str, class_str2int=target_manager.class_str2int)
        """
        for key, value in kwargs.items():
            if hasattr(self._model_result.derived_values, key):
                setattr(self._model_result.derived_values, key, value)
            else:
                assert False, f"Unknown derived value field: {key}"

        self._notify_observers(
            BiotrainerModelUpdate(current_model_result=self._model_result)  # TODO
        )

    # ============= Training Results =============
    def add_training_iteration(self, split_name: str, epoch_metrics: EpochMetrics) -> None:
        if split_name not in self._model_result.training_results:
            self._model_result.training_results[split_name] = TrainingResult()

        result = self._model_result.training_results[split_name]
        result.training_losses.append(epoch_metrics.training["loss"])
        result.validation_losses.append(epoch_metrics.validation["loss"])

        # Update best epoch if needed
        # TODO?
        if (result.best_epoch_metrics is None or
                epoch_metrics.validation["loss"] < result.best_epoch_metrics.validation["loss"]):
            result.best_epoch_metrics = epoch_metrics

        self._notify_observers(
            BiotrainerModelUpdate(current_model_result=self._model_result,
                                  training_iteration=(split_name, epoch_metrics))
        )

    def update_training_result(self, split_name: str, **kwargs) -> None:
        """Update training result fields for a split.

        Example:
            output_manager.update_training_result(
                "split_0",
                n_training_ids=100,
                split_hyper_params={"lr": 0.001}
            )
        """
        if split_name not in self._model_result.training_results:
            self._model_result.training_results[split_name] = TrainingResult()

        result = self._model_result.training_results[split_name]
        for key, value in kwargs.items():
            if hasattr(result, key):
                setattr(result, key, value)
            else:
                assert False, f"Unknown training result field: {key}"

        self._notify_observers(
            BiotrainerModelUpdate(current_model_result=self._model_result)
        )

    # ============= Test Results =============
    def add_test_result(self, test_set_id: str, **kwargs) -> None:
        """Add or update test results.

        Example:
            output_manager.add_test_result(
                "test_set_1",
                inference_result=BiotrainerInferenceResult(),
                bootstrapped_metrics=[...]
            )
        """
        if test_set_id not in self._model_result.test_results:
            self._model_result.test_results[test_set_id] = TestResult()

        result = self._model_result.test_results[test_set_id]
        for key, value in kwargs.items():
            if hasattr(result, key):
                setattr(result, key, value)
            else:
                assert False, f"Unknown test result field: {key}"

        self._notify_observers(
            BiotrainerModelUpdate(current_model_result=self._model_result)
        )

    # ============= Predictions =============
    def add_predictions(self, predictions: List[BiotrainerPrediction]) -> None:
        if self._model_result.predictions:
            assert False, "Predictions already set!"

        self._model_result.predictions = predictions
        self._notify_observers(BiotrainerModelUpdate(current_model_result=self._model_result))

    # ============= Access & Serialization =============
    @property
    def model_result(self) -> BiotrainerModelResult:
        """Read-only access to the underlying model result."""
        return self._model_result

    def maybe_load_existing_result(self, output_dir: Path, model_hash: str) -> bool:
        if output_dir.exists():
            try:
                file_path = output_dir / "out.yml"
                if file_path.exists():
                    loaded_result = BiotrainerModelResult.from_file(file_path)
                    if loaded_result.derived_values.model_hash != model_hash:
                        raise AssertionError(f"Model hash mismatch: "
                                             f"{loaded_result.derived_values.model_hash} != {model_hash},"
                                             f"this should have been caught earlier in the pipeline!")
                    self._model_result = loaded_result
                    return True
            except Exception as e:
                logger.error(f"Error while loading existing output file: {output_dir / 'out.yml'}: {str(e)}")
        return False

    def write_to_file(self, output_dir: Path) -> None:
        """Write results to YAML file."""
        output_dict = self._model_result.model_dump(exclude_none=True)

        with open(output_dir / "out.yml", "w") as f:
            yaml.dump(
                output_dict,
                f,
                Dumper=yaml.RoundTripDumper,
                default_flow_style=False
            )


class InferenceOutputManager(OutputManager):
    def __init__(self,
                 training_result: BiotrainerModelResult,
                 output_file_path: Optional[Path] = None,
                 automatic_path_correction: bool = True):
        super().__init__(observers=[])
        self._input_config = training_result.config
        self._derived_values = training_result.derived_values
        self._training_results = training_result.training_results
        self._test_results = training_result.test_results
        self._predictions = training_result.predictions

        if automatic_path_correction:
            self._do_automatic_path_correction(output_file_path)

        if self._derived_values.biotrainer_version != __version__:
            print("WARNING: The loaded model was trained on a different biotrainer version than currently running.\n"
                  "This may lead to unexpected behaviour if another torch version was used for training.")

    def _do_automatic_path_correction(self, output_file_path: Optional[Path]):
        if output_file_path is None:
            return

        log_dir = self._input_config["log_dir"]
        log_dir_path = Path(log_dir)
        if not log_dir_path.exists():
            # Split the output file path and reconstruct without the last component
            output_dir = Path(*output_file_path.parts[:-1])

            new_log_dir_path = output_dir
            if not new_log_dir_path.exists():
                print(f"Could not automatically correct the checkpoint file paths! "
                      f"Tried: {str(new_log_dir_path)} but it does not exist.")
            elif len(os.listdir(str(new_log_dir_path))) == 0:
                print(f"Found corrected path ({str(new_log_dir_path)}), but it does not contain any files!")
            else:
                print(f"Reading checkpoint(s) from directory: {new_log_dir_path}..")
                self._input_config["log_dir"] = new_log_dir_path

    def protocol(self):
        return Protocol.from_string(self._input_config["protocol"])

    def embedder_name(self):
        return self._input_config["embedder_name"]

    def use_half_precision(self):
        return self._input_config["use_half_precision"]

    def log_dir(self):
        return self._input_config["log_dir"]

    def device(self):
        # The stored device records what the training run used, not what the caller is asking for now, so a model
        # trained on cuda has to stay loadable on a machine without one.
        stored_device = self._input_config["device"]
        try:
            return get_device(stored_device)
        except ValueError:
            device = get_device()
            logger.warning(f"Model was trained on device {stored_device}, which is not available here. "
                           f"Using {device} instead.")
            return device

    def dimension_reduction_method(self):
        return self._input_config.get("dimension_reduction_method", None)

    def n_reduced_components(self):
        return self._input_config.get("n_reduced_components", None)

    def disable_pytorch_compile(self):
        return self._input_config["disable_pytorch_compile"]

    def n_features(self):
        return self._derived_values.n_features

    def class_int2str(self):
        return self._derived_values.class_int2str

    def class_str2int(self):
        return self._derived_values.class_str2int

    def training_results(self):
        return self._training_results

    def split_config(self, split_name: str):
        config = {**self._input_config, **self._derived_values.model_dump()}
        config.update(self._training_results[split_name].split_hyper_params)
        return deepcopy(config)

    def adapter_path(self) -> Optional[Path]:
        if "finetuning_config" in self._input_config:
            finetuning_path = Path(self._input_config["log_dir"])
            if finetuning_path.exists():
                return finetuning_path
            raise FileNotFoundError(f"Could not find finetuning checkpoint at {finetuning_path}")
        return None

    def feature_scaler(self) -> Optional[FeatureScaler]:
        scaling_method = self._input_config.get("scaling_method", None)
        if scaling_method is not None:
            file_name = FeatureScaler.get_file_name(scaling_method)
            scaler_path = Path(self._input_config["log_dir"]) / file_name
            if scaler_path.exists():
                feature_scaler = FeatureScaler.load(method=scaling_method,
                                                    protocol=self.protocol(),
                                                    load_path=scaler_path)
                return feature_scaler
            raise FileNotFoundError(f"Could not find feature scaling checkpoint at {scaler_path}")
        return None

    def class_weights(self):
        class_weights = self._derived_values.computed_class_weights
        # Restore sorting of class weights (are sorted by ascending index from class_int2str)
        if class_weights is not None:
            class_weights = torch.tensor([class_weights[idx] for idx in range(len(class_weights))])
        return class_weights


""" 
TODO Is removing split ids still necessary?
#with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_output_path = tmp_dir_name + "/tmp_output.yml"
            with open(out_file_path, "r") as output_file, open(tmp_output_path, "w") as tmp_output_file:
                ids_list = False
                for line in output_file.readlines():
                    if line.strip() == "training_ids:" or line.strip() == "validation_ids:":
                        ids_list = True
                        continue
                    elif ids_list and ("-" in line and ":" not in line):
                        continue
                    else:
                        ids_list = False
                    if not ids_list:
                        tmp_output_file.write(line)
"""
