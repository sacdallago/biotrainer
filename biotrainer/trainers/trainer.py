from typing import Dict, Any, Optional

from .default_pipeline import DefaultPipeline
from .pipeline import Pipeline, PipelineContext

from ..output_files import OutputManager


class Trainer:

    def __init__(self,
                 config: Dict[str, Any],
                 output_manager: OutputManager,
                 custom_pipeline: Optional[Pipeline] = None):
        self.pipeline_context = PipelineContext(config=config,
                                                output_manager=output_manager,
                                                custom_pipeline=False)
        self.pipeline: Pipeline = custom_pipeline if custom_pipeline else DefaultPipeline(config).pipeline

    def run(self) -> OutputManager:
        pipeline_context = self.pipeline.execute(context=self.pipeline_context)
        return pipeline_context.output_manager

