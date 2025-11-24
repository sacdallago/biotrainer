from pathlib import Path

from ..pipeline import PipelineStep, PipelineContext
from ..pipeline.pipeline_step import PipelineStepType

from ...utilities import FeatureScaler, get_logger

logger = get_logger(__name__)


class ScalingStep(PipelineStep):
    def get_step_type(self) -> PipelineStepType:
        return PipelineStepType.SCALING

    def process(self, context: PipelineContext) -> PipelineContext:
        id2emb = context.id2emb
        target_manager = context.target_manager
        assert id2emb is not None and len(id2emb) > 0, f"id2emb cannot be None or empty at the scaling step!"
        assert target_manager is not None, f"target_manager cannot be None at the scaling step!"
        old_n_embeddings = len(id2emb)
        scaling_method = context.config.get("scaling_method", "none")

        if scaling_method != "none":
            training_ids = set(target_manager.training_ids)
            training_embs = {seq_id: embd for seq_id, embd in id2emb.items() if seq_id in training_ids}
            other_embs = {seq_id: embd for seq_id, embd in id2emb.items() if seq_id not in training_ids}

            # Fit on training embeddings
            feature_scaler = FeatureScaler(method=scaling_method)
            feature_scaler = feature_scaler.fit(training_embs)

            # Transform all embeddings
            training_embs_scaled = feature_scaler.transform(training_embs)
            other_embs_scaled = feature_scaler.transform(other_embs)
            id2emb = {**training_embs_scaled, **other_embs_scaled}
            # Save fitted scaler
            save_dir = context.config["log_dir"]
            scaling_save_name = f"{scaling_method}_scaling.pkl"
            feature_scaler.save(Path(save_dir) / scaling_save_name)
            logger.info(f"Fitted feature scaling {scaling_method}!")
        else:
            logger.info(f"No feature scaling performed (as configured).")

        assert old_n_embeddings == len(id2emb), f"The number of embeddings changed during feature scaling!"
        context.id2emb = id2emb

        return context