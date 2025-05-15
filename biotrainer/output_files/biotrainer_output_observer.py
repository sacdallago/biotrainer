from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple

from ..utilities import EpochMetrics


@dataclass
class OutputData:
    config: Optional[Dict[str, Any]] = None
    derived_values: Optional[Dict[str, Any]] = None
    split_specific_values: Optional[Dict[str, Any]] = None
    training_iteration: Optional[Tuple[str, EpochMetrics]] = None
    test_results: Optional[Dict[str, Any]] = None
    predictions: Optional[Dict[str, Any]] = None


class BiotrainerOutputObserver(ABC):
    @abstractmethod
    def update(self, data: OutputData) -> None:
        """Handle an output event with associated data."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleanup resources when training is finished."""
        pass