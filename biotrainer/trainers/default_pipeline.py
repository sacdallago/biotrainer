from typing import Optional

from .pipeline import Pipeline, PipelineStep
from .steps import SetupStep, EmbeddingStep, ProjectionStep, DataLoadingStep, TrainingStep, TestingStep, PostProcessStep, InputValidationStep


class DefaultPipeline:
    def __init__(self):
        self.pipeline = (Pipeline()
                         .add_step(SetupStep())
                         .add_step(InputValidationStep())
                         .add_step(EmbeddingStep())
                         .add_step(ProjectionStep())
                         .add_step(DataLoadingStep())
                         .add_step(TrainingStep())
                         .add_step(TestingStep())
                         .add_step(PostProcessStep())
                         )

    def with_custom_steps(self,
                          custom_setup_step: Optional[PipelineStep] = None,
                          custom_input_validation_step: Optional[PipelineStep] = None,
                          custom_embedding_step: Optional[PipelineStep] = None,
                          custom_projection_step: Optional[PipelineStep] = None,
                          custom_data_loading_step: Optional[PipelineStep] = None,
                          custom_training_step: Optional[PipelineStep] = None,
                          custom_testing_step: Optional[PipelineStep] = None,
                          custom_postprocess_step: Optional[PipelineStep] = None):

        custom_steps = [
            custom_setup_step,
            custom_input_validation_step,
            custom_embedding_step,
            custom_projection_step,
            custom_data_loading_step,
            custom_training_step,
            custom_testing_step,
            custom_postprocess_step
        ]

        for custom_step in custom_steps:
            if custom_step is not None:
                self.pipeline.replace_step(custom_step)

        return self
