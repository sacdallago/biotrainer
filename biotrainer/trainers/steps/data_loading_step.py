from ..pipeline import PipelineContext, PipelineStep
from ..pipeline.pipeline_step import PipelineStepType

from ..target_manager import TargetManager

from ...utilities import get_logger

logger = get_logger(__name__)


class DataLoadingStep(PipelineStep):

    def get_step_type(self) -> PipelineStepType:
        return PipelineStepType.DATA_LOADING

    def process(self, context: PipelineContext) -> PipelineContext:
        id2emb = context.id2emb
        assert id2emb is not None and len(id2emb) > 0, f"id2emb cannot be None or empty at the data loading step!"

        # Load TARGETS and SETS from input data
        target_manager = TargetManager(protocol=context.config["protocol"], input_data=context.input_data,
                                       ignore_file_inconsistencies=context.config["ignore_file_inconsistencies"],
                                       cross_validation_method=context.config["cross_validation_config"]["method"],
                                       interaction=context.config.get("interaction"))
        target_manager.load()

        context.target_manager = target_manager
        return context
