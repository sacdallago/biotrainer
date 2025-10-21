import pickle

from pathlib import Path
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
        target_manager = context.target_manager
        assert id2emb is not None and len(id2emb) > 0, f"id2emb cannot be None or empty at the projection step!"
        assert target_manager is not None, f"target_manager cannot be None at the projection step!"
        old_n_embeddings = len(id2emb)

        dimension_reduction_method = context.config.get("dimension_reduction_method")
        n_reduced_components = context.config.get("n_reduced_components")

        if self._is_dimension_reduction_possible(context, dimension_reduction_method, n_reduced_components, id2emb):
            training_ids = set(target_manager.training_ids)
            training_embs = {seq_id: embd for seq_id, embd in id2emb.items() if seq_id in training_ids}
            other_embs = {seq_id: embd for seq_id, embd in id2emb.items() if seq_id not in training_ids}
            training_embs_reduced, fitted_transform = EmbeddingService.embeddings_dimensionality_reduction(
                embeddings=training_embs,
                dimension_reduction_method=dimension_reduction_method,
                n_reduced_components=n_reduced_components)
            other_embs_reduced, _ = EmbeddingService.embeddings_dimensionality_reduction(
                embeddings=other_embs,
                dimension_reduction_method=dimension_reduction_method,
                n_reduced_components=n_reduced_components,
                fitted_transform=fitted_transform
            )
            # Combine embeddings
            id2emb = {**training_embs_reduced, **other_embs_reduced}
            # Save fitted transform
            save_dir = context.config["log_dir"]
            transform_save_name = f"{dimension_reduction_method}_{n_reduced_components}_transform.pkl"
            with open(Path(save_dir) / transform_save_name, "wb") as f:
                pickle.dump(fitted_transform, f)
        else:
            logger.info(f"No dimension reduction performed (as configured).")

        assert old_n_embeddings == len(id2emb), f"The number of embeddings changed during dimensionality reduction!"
        context.id2emb = id2emb

        logger.info(f"Finished embeddings dimensionality reduction!")
        return context
