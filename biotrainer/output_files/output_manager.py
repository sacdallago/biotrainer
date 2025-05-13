import torch
import functools

from enum import Enum
from ruamel import yaml
from pathlib import Path
from typing import Dict, Callable, Any

from ..protocols import Protocol
from ..utilities import get_logger

logger = get_logger(__name__)


class OutputManagerEvent(Enum):
    GENERIC = 0
    TRAINING_ITERATION = 1


def notify_callbacks(event_type: OutputManagerEvent = OutputManagerEvent.GENERIC):
    """
    Decorator that notifies all registered callbacks after a method executes.

    Args:
        event_type: Optional custom event type. If None, uses the method name.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Execute the original method
            result = func(self, *args, **kwargs)

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(event_type, self._data)
                except Exception as e:
                    logger.error(f"Error in callback during {event_type}: {str(e)}")

            # Return the original result
            return result

        return wrapper

    return decorator


class OutputManager:
    """Manages training outputs, results, and logging in a structured way."""

    def __init__(self, config: Dict[str, Any], model_hash: str):
        self._input_config = {str(k): self._convert_config_value(v) for k, v in config.items()}
        self._derived_values = {"model_hash": model_hash}
        self._split_specific_values = {}
        self._training_results = {}
        self._test_results = {}
        self._predictions: Dict[str, Any] = {}  # seq_id -> prediction

        self._callbacks = []

    @staticmethod
    def _convert_config_value(value: Any) -> Any:
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, Protocol):
            return value.name
        return value

    def add_callback(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    @notify_callbacks(event_type=OutputManagerEvent.GENERIC)
    def add_derived_values(self, derived_values: Dict[str, Any]):
        self._derived_values.update(derived_values)

    @notify_callbacks(event_type=OutputManagerEvent.GENERIC)
    def add_split_specific_values(self, split_specific_values: Dict[str, Any],
                                  split_name: str,
                                  ):
        if split_name not in self._split_specific_values:
            self._split_specific_values[split_name] = {}
        self._split_specific_values[split_name].update(split_specific_values)

    @notify_callbacks(event_type=OutputManagerEvent.TRAINING_ITERATION)
    def add_training_iteration(self, epoch_metrics: Dict[str, Any],
                               split_name: str,
                               ):
        if split_name not in self._training_results:
            self._training_results[split_name] = []

        self._training_results[split_name].append(epoch_metrics)

    @notify_callbacks(event_type=OutputManagerEvent.TRAINING_ITERATION)
    def add_test_set_result(self, test_set_id: str, test_set_results: Dict[str, Any]):
        if test_set_id not in self._test_results:
            self._test_results[test_set_id] = {}
        self._test_results[test_set_id].update(test_set_results)

    @notify_callbacks(event_type=OutputManagerEvent.TRAINING_ITERATION)
    def add_prediction_result(self, prediction_results: Dict[str, Any]):
        assert len(self._predictions) == 0, f"Tried to add predictions more than one time!"
        self._predictions.update(prediction_results)

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
