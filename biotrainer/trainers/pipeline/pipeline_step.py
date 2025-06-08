from abc import ABC, abstractmethod

from .pipeline_context import PipelineContext


class PipelineStep(ABC):

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__