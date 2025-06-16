from typing import Dict, Any

from ..pipeline import PipelineContext, PipelineStep
from ..pipeline.pipeline_step import PipelineStepType

from ...protocols import Protocol
from ...utilities import get_logger
from ...embedders import EmbeddingService

logger = get_logger(__name__)


class ProjectionStep(PipelineStep):

    def get_step_type(self) -> PipelineStepType:
        return PipelineStepType.PROJECTION

    @staticmethod
    def _is_dimension_reduction_possible(context: PipelineContext, dimension_reduction_method, n_reduced_components,
                                         id2emb: Dict[str, Any]) -> bool:
        protocol: Protocol = context.config["protocol"]

        min_number_embeddings = 3
        min_number_dimensions = 3

        number_embeddings = len(id2emb)
        number_dimensions = next(iter(id2emb.values())).shape[0]
        if (protocol.using_per_sequence_embeddings() and dimension_reduction_method and n_reduced_components and
                number_embeddings >= min_number_embeddings and
                number_dimensions >= min_number_dimensions):
            return True
        else:
            if dimension_reduction_method and n_reduced_components:
                # Check for errors
                if number_embeddings < min_number_embeddings:
                    raise ValueError(f"Dimensionality reduction cannot be performed as \
                                the number of samples is less than {min_number_embeddings}")
                if number_dimensions < 3:
                    raise ValueError(f"Dimensionality reduction cannot be performed as \
                                the original embedding dimension is less than {min_number_dimensions}")
                if not protocol.using_per_sequence_embeddings():
                    raise ValueError(f"Dimensionality reduction cannot be performed as \
                                the embeddings are not per-sequence embeddings")
            return False

    def process(self, context: PipelineContext) -> PipelineContext:
        id2emb = context.id2emb
        assert id2emb is not None and len(id2emb) > 0, f"id2emb cannot be None or empty at the projection step!"

        dimension_reduction_method = context.config.get("dimension_reduction_method")
        n_reduced_components = context.config.get("n_reduced_components")

        if self._is_dimension_reduction_possible(context, dimension_reduction_method, n_reduced_components, id2emb):
            id2emb = EmbeddingService.embeddings_dimensionality_reduction(
                embeddings=id2emb,
                dimension_reduction_method=dimension_reduction_method,
                n_reduced_components=n_reduced_components)
        else:
            logger.info(f"No dimension reduction performed (as configured).")

        context.id2emb = id2emb
        return context
