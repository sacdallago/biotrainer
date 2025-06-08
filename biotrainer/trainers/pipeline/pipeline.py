from typing import List

from .pipeline_step import PipelineStep
from .pipeline_context import PipelineContext

from ...utilities import get_logger

logger = get_logger(__name__)

class Pipeline:
    def __init__(self):
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        self.steps.append(step)
        return self  # For method chaining

    def execute(self, context: PipelineContext) -> PipelineContext:
        current_context = context
        for step in self.steps:
            try:
                logger.info(f"Executing step: {step.__class__.__name__}")  # Setup step is not logged on purpose
                current_context = step.process(current_context)
            except Exception as e:
                logger.error(f"Error in step {step.__class__.__name__}: {str(e)}")
                raise

        return current_context
