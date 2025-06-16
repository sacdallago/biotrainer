import time
import datetime

from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Any

from .training_factory import TrainingFactory

from ..cv_splitter import CrossValidationSplitter
from ..pipeline import PipelineContext, PipelineStep

from ...solvers import Solver
from ...models import count_parameters
from ...utilities import get_logger, Split, SplitResult, EpochMetrics, METRICS_WITHOUT_REVERSED_SORTING

logger = get_logger(__name__)


class PostProcessStep(PipelineStep):
    @staticmethod
    def _onnx_export(context: PipelineContext):
        try:
            context.best_split.solver.save_as_onnx(embedding_dimension=context.n_features)
        except Exception as e:
            logger.error("Could not save model as ONNX!")

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

        # LOG TIME
        self._log_time(context)

        return context
