import torch

from ruamel import yaml
from pathlib import Path
from typing import Dict, Any, List

from .biotrainer_output_observer import BiotrainerOutputObserver, OutputData

from ..utilities import EpochMetrics
from ..protocols import Protocol
from ..utilities import get_logger

logger = get_logger(__name__)


class OutputManager:
    """Manages training outputs, results, and logging in a structured way."""

    def __init__(self, observers: List[BiotrainerOutputObserver]):
        self._observers: List[BiotrainerOutputObserver] = observers

        self._input_config = {}
        self._derived_values = {}
        self._split_specific_values = {}
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
    def _convert_config_value(value: Any) -> Any:
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, Protocol):
            return value.name
        return value

    def add_config(self, config: Dict[str, Any]) -> None:
        self._input_config = {str(k): self._convert_config_value(v) for k, v in config.items()}

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

    def write_to_file(self, output_dir: Path) -> Dict[str, Any]:
        output_result = {"config": self._sort_dict(self._input_config),
                         "derived_values": self._sort_dict(self._derived_values),
                         "splits": self._sort_dict(self._split_specific_values),
                         "test_results": self._sort_dict(self._test_results),
                         "predictions": self._sort_dict(self._predictions),
                         }
        dumper = yaml.RoundTripDumper
        with open(output_dir / "out.yml", "w") as f:
            f.write(
                yaml.dump(output_result, Dumper=dumper, default_flow_style=False)
            )

        return output_result
