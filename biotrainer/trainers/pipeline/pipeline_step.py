from enum import Enum
from abc import ABC, abstractmethod

from .pipeline_context import PipelineContext


class PipelineStepType(Enum):
    SETUP = "setup"
    INPUT_VALIDATION = "input_validation"
    EMBEDDING = "embedding"
    DATA_LOADING = "data_loading"
    PROJECTION = "projection"
    DATASET_CREATION = "dataset_creation"
    TRAINING = "training"
    TESTING = "testing"
    POST_PROCESS = "post_process"
    CUSTOM = "custom"


class PipelineStep(ABC):

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        pass

    @abstractmethod
    def get_step_type(self) -> PipelineStepType:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__
