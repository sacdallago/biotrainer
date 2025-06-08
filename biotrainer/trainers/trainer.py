from typing import Dict, Any, Optional

from .pipeline import Pipeline, PipelineContext
from .steps import SetupStep, TrainingStep, TestingStep, ProjectionStep, PostProcessStep, EmbeddingStep, DataLoadingStep

from ..output_files import OutputManager


class Trainer:

    def __init__(self, config: Dict[str, Any], output_manager: OutputManager,
                 custom_pipeline: Optional[Pipeline] = None):
        self.pipeline_context = PipelineContext(config=config,
                                                output_manager=output_manager,
                                                custom_pipeline=False)
        self.pipeline = custom_pipeline if custom_pipeline else self._default_pipeline()

    @staticmethod
    def _default_pipeline():
        return (Pipeline()
                         .add_step(SetupStep())
                         .add_step(EmbeddingStep())
                         .add_step(ProjectionStep())
                         .add_step(DataLoadingStep())
                         .add_step(TrainingStep())
                         .add_step(TestingStep())
                         .add_step(PostProcessStep())
                         )

    def run(self) -> OutputManager:
        pipeline_context = self.pipeline.execute(context=self.pipeline_context)
        return pipeline_context.output_manager

