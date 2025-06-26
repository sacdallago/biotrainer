from ..pipeline import PipelineStep, PipelineContext
from ..pipeline.pipeline_step import PipelineStepType

from ...validations import InputValidator


class InputValidationStep(PipelineStep):
    def get_step_type(self) -> PipelineStepType:
        return PipelineStepType.INPUT_VALIDATION

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.config.get("validate_input", True):
            protocol = context.config["protocol"]
            input_validator = InputValidator(protocol=protocol)
            validated_input_data = input_validator.validate(context.input_data)
            # No errors - set validated input data as input data
            context.input_data = validated_input_data
        return context