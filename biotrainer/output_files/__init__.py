from pathlib import Path
from typing import Dict, Any, List, Union

from .tensorboard_writer import TensorboardWriter
from .biotrainer_output_observer import BiotrainerOutputObserver
from .output_manager import OutputManager, InferenceOutputManager, OutputData


def output_observer_factory(output_dir: Union[Path, str], config: Dict[str, Any]) -> List[BiotrainerOutputObserver]:
    result = []
    if "external_writer" in config:
        external_writer: str = config["external_writer"]
        if external_writer.lower() == "none":
            return []
        if external_writer.lower() == "tensorboard":
            result.append(TensorboardWriter(log_dir=Path(output_dir) / "runs"))
    return result


__all__ = ["OutputManager", "OutputData", "InferenceOutputManager", "BiotrainerOutputObserver",
           "output_observer_factory"]
