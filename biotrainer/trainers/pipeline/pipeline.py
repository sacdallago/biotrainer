from typing import Dict, List

from .pipeline_context import PipelineContext
from .pipeline_step import PipelineStep, PipelineStepType

from ...utilities import get_logger

logger = get_logger(__name__)


class Pipeline:
    def __init__(self):
        self.steps_dict: Dict[PipelineStepType, PipelineStep] = {}
        self.step_order: List[PipelineStepType] = []

    def add_step(self, step: PipelineStep):
        step_type = step.get_step_type()

        self.steps_dict[step_type] = step
        if step_type not in self.step_order:
            self.step_order.append(step_type)
        return self

    def replace_step(self, step: PipelineStep):
        step_type = step.get_step_type()

        if step_type not in self.steps_dict:
            raise ValueError(f"Step type {step_type} not found in pipeline")
        self.steps_dict[step_type] = step
        return self

    def execute(self, context: PipelineContext) -> PipelineContext:
        current_context = context
        for step_type in self.step_order:
            step = self.steps_dict[step_type]
            try:
                logger.info(f"Executing step: {step.__class__.__name__}")
                current_context = step.process(current_context)
            except Exception as e:
                logger.error(f"Error in step {step.__class__.__name__}: {str(e)}")
                raise
        return current_context
