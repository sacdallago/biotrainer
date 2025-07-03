import time
import datetime

from ..pipeline import PipelineContext, PipelineStep
from ..pipeline.pipeline_step import PipelineStepType

from ...utilities import get_logger

logger = get_logger(__name__)


class PostProcessStep(PipelineStep):

    def get_step_type(self) -> PipelineStepType:
        return PipelineStepType.POST_PROCESS

    @staticmethod
    def _onnx_export(context: PipelineContext):
        try:
            context.best_split.solver.save_as_onnx(embedding_dimension=context.n_features)
        except Exception:
            logger.error("Could not save model as ONNX!")

    @staticmethod
    def _save_finetuned_model(context: PipelineContext):
        embedding_service = context.embedding_service
        if embedding_service:
            embedding_service.save_embedder(output_dir=context.config["log_dir"])

    @staticmethod
    def _log_time(context: PipelineContext):
        pipeline_end_time = time.perf_counter()
        pipeline_end_time_abs = str(datetime.datetime.now().isoformat())
        pipeline_elapsed_time = pipeline_end_time - context.pipeline_start_time
        logger.info(f"Pipeline end time: {pipeline_end_time_abs}")
        logger.info(f"Total elapsed time for pipeline: {pipeline_elapsed_time} [s]")
        context.output_manager.add_derived_values({'pipeline_end_time': pipeline_end_time_abs})
        context.output_manager.add_derived_values({'pipeline_elapsed_time': pipeline_elapsed_time})

        logger.info(f"Extensive output information can be found at {context.config['output_dir']}/out.yml")

    def process(self, context: PipelineContext) -> PipelineContext:
        assert context.best_split is not None, f"Best split cannot be None at postprocess step!"

        # SAVE BEST SPLIT AS ONNX
        self._onnx_export(context)

        # SAVE FINETUNED EMBEDDER MODEL (IF APPLICABLE)
        self._save_finetuned_model(context)

        # LOG TIME
        self._log_time(context)

        return context
