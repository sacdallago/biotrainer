import os
import torch

from ruamel import yaml
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, List, Union

from .biotrainer_output_observer import BiotrainerOutputObserver, OutputData

from ..protocols import Protocol
from ..utilities import EpochMetrics, get_device, get_logger, __version__

logger = get_logger(__name__)


class OutputManager:
    """Manages training outputs, results, and logging in a structured way."""

    def __init__(self, observers: List[BiotrainerOutputObserver]):
        self._observers: List[BiotrainerOutputObserver] = observers

        self._input_config = {}
        self._derived_values = {}
        self._split_specific_values = {}  # TODO Refactor into _training_results
        self._training_results: Dict[str, List[EpochMetrics]] = {}  # split_name -> Epoch Metrics
        self._test_results = {}
        self._predictions: Dict[str, Any] = {}  # seq_id -> prediction

    def _notify_observers(self, data: OutputData) -> None:
        for observer in self._observers:
            try:
                observer.update(data)
            except Exception as e:
                logger.error(f"Error in observer during output event: {str(e)}")

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
        self._input_config = {str(k): self._convert_config_value(k, v) for k, v in
                              config.items()}

        self._notify_observers(data=OutputData(config=self._input_config))

    def add_derived_values(self, derived_values: Dict[str, Any]):
        self._derived_values.update(derived_values)

        self._notify_observers(data=OutputData(derived_values=self._derived_values))

    def add_split_specific_values(self, split_name: str, split_specific_values: Dict[str, Any]):
        if split_name not in self._split_specific_values:
            self._split_specific_values[split_name] = {}
        self._split_specific_values[split_name].update(split_specific_values)

        self._notify_observers(data=OutputData(split_specific_values=self._split_specific_values))

    def add_training_iteration(self, split_name: str, epoch_metrics: EpochMetrics):
        if split_name not in self._training_results:
            self._training_results[split_name] = []
        self._training_results[split_name].append(epoch_metrics)

        self._notify_observers(data=OutputData(training_iteration=(split_name, epoch_metrics)))

    def add_test_set_result(self, test_set_id: str, test_set_results: Dict[str, Any]):
        if test_set_id not in self._test_results:
            self._test_results[test_set_id] = {}
        self._test_results[test_set_id].update(test_set_results)

        self._notify_observers(data=OutputData(test_results=self._test_results))

    def add_prediction_result(self, prediction_results: Dict[str, Any]):
        assert len(self._predictions) == 0, f"Tried to add predictions more than one time!"
        self._predictions.update(prediction_results)

        self._notify_observers(data=OutputData(predictions=self._predictions))

    @staticmethod
    def _sort_dict(d: dict):
        return dict(sorted(d.items(), key=lambda t: t[0]))

    def _format_splits_with_results(self) -> Dict[str, Any]:
        result = deepcopy(self._split_specific_values)
        for split_name, epoch_metrics in self._training_results.items():
            result[split_name].update({
                "training_loss": {},
                "validation_loss": {}
            })
            for epoch_metric in epoch_metrics:
                epoch_str = str(epoch_metric.epoch)
                result[split_name]["training_loss"][epoch_str] = epoch_metric.training["loss"]
                result[split_name]["validation_loss"][epoch_str] = epoch_metric.validation["loss"]
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"config": self._sort_dict(self._input_config),
                "database_type": "PPI" if self._input_config.get("interaction") is not None else "Protein",
                "derived_values": self._sort_dict(self._derived_values),
                "training_results": self._sort_dict(self._format_splits_with_results()),
                "test_results": self._sort_dict(self._test_results),
                "predictions": self._sort_dict(self._predictions),
                }

    def write_to_file(self, output_dir: Path) -> Dict[str, Any]:
        output_result = self.to_dict()
        dumper = yaml.RoundTripDumper
        with open(output_dir / "out.yml", "w") as f:
            f.write(
                yaml.dump(output_result, Dumper=dumper, default_flow_style=False)
            )

        return output_result


class InferenceOutputManager(OutputManager):
    def __init__(self, output_file_path: Path, automatic_path_correction: bool = True):
        super().__init__(observers=[])
        print(f"Reading {output_file_path}..")
        with open(output_file_path, "r") as output_file:
            training_output = yaml.load(output_file, Loader=yaml.RoundTripLoader)
            self._input_config = training_output["config"]
            self._derived_values = training_output["derived_values"]
            self._training_results = training_output["training_results"]
            self._test_results = training_output["test_results"]
            self._predictions = training_output["predictions"]

        if automatic_path_correction:
            self._do_automatic_path_correction(output_file_path)

        if self._derived_values["biotrainer_version"] != __version__:
            print("WARNING: The loaded model was trained on a different biotrainer version than currently running.\n"
                  "This may lead to unexpected behaviour if another torch version was used for training.")

    def _do_automatic_path_correction(self, output_file_path: Path):
        log_dir = self._input_config["log_dir"]
        log_dir_path = Path(log_dir)
        if not log_dir_path.exists():
            # Expect checkpoints to be in output/model_choice/embedder_name
            checkpoints_path = Path(self._input_config["model_choice"]) / \
                               self._input_config["embedder_name"].split("/")[-1]
            # Split the output file path and reconstruct without the last component
            output_dir = Path(*output_file_path.parts[:-1])

            new_log_dir_path = output_dir / checkpoints_path
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
        return get_device(self._input_config["device"])

    def dimension_reduction_method(self):
        return self._input_config.get("dimension_reduction_method", None)

    def n_reduced_components(self):
        return self._input_config.get("n_reduced_components", None)

    def disable_pytorch_compile(self):
        return self._input_config["disable_pytorch_compile"]

    def n_features(self):
        return self._derived_values["n_features"]

    def class_int2str(self):
        return self._derived_values.get("class_int2str", None)

    def class_str2int(self):
        return self._derived_values.get("class_str2int", None)

    def training_results(self):
        return self._training_results

    def split_config(self, split_name: str):
        config = {**self._input_config, **self._derived_values}
        config.update(self._training_results[split_name]["split_hyper_params"])
        return deepcopy(config)

    def adapter_path(self) -> Union[Path, None]:
        if "finetuning_config" in self._input_config:
            finetuning_path = Path(self._input_config["log_dir"])
            if finetuning_path.exists():
                return finetuning_path
        return None

    def class_weights(self):
        class_weights = self._derived_values.get("computed_class_weights", None)
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
