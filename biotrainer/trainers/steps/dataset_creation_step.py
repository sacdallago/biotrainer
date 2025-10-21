import torch

from typing import Union
from copy import deepcopy

from ..pipeline import PipelineContext, PipelineStep
from ..pipeline.pipeline_step import PipelineStepType

from ..target_manager import TargetManager

from ...protocols import Protocol
from ...utilities import get_logger, EmbeddingDatasetSample

logger = get_logger(__name__)


class DatasetCreationStep(PipelineStep):

    def get_step_type(self) -> PipelineStepType:
        return PipelineStepType.DATASET_CREATION

    @staticmethod
    def _get_class_weights(context: PipelineContext, target_manager: TargetManager) -> Union[
        None, torch.FloatTensor]:
        protocol = context.config["protocol"]
        # Get x_to_class specific logs and weights
        class_weights = None
        if protocol in Protocol.classification_protocols():
            context.output_manager.add_derived_values(
                {'class_int2str': target_manager.class_int2str,
                 'class_str2int': target_manager.class_str2int}
            )
            context.class_str2int = target_manager.class_str2int

            # Compute class weights to pass as bias to model if option is set
            class_weights = target_manager.compute_class_weights()
            if class_weights is not None:
                computed_class_weights = {class_index: class_value.item() for
                                          class_index, class_value in enumerate(class_weights)}
                context.output_manager.add_derived_values({'computed_class_weights': computed_class_weights})

        return class_weights

    def process(self, context: PipelineContext) -> PipelineContext:
        id2emb = context.id2emb
        assert id2emb is not None and len(id2emb) > 0, f"id2emb cannot be None or empty at the dataset creation step!"
        assert context.target_manager is not None, f"target_manager cannot be None at the dataset creation step!"

        # TARGETS => DATASETS
        target_manager = context.target_manager
        first_embedding = next(iter(context.id2emb.values()))

        finetuning = "finetuning_config" in context.config
        if finetuning:
            train_dataset, val_dataset, test_datasets, prediction_dataset = target_manager.get_sequence_datasets()
            baseline_test_datasets = {test_name: [
                EmbeddingDatasetSample(seq_id=data_point.seq_id, embedding=context.id2emb[data_point.seq_id],
                                       target=data_point.target) for data_point in test_data] for test_name, test_data
                                      in test_datasets.items()}
            random_masking = context.config.get("finetuning_config", {}).get("random_masking", False)
            if random_masking:
                logger.info("Random masking is enabled. All targets will be overwritten with masked sequence targets.")
        else:
            train_dataset, val_dataset, test_datasets, prediction_dataset = target_manager.get_embedding_datasets(
                context.id2emb)
            baseline_test_datasets = deepcopy(test_datasets)

        # LOG COMMON VALUES FOR ALL k-fold SPLITS:
        context.n_features = first_embedding.shape[-1]  # Last position in shape is always embedding dim
        context.n_classes = target_manager.number_of_outputs

        logger.info(f"Number of features: {context.n_features}")
        logger.info(f"Number of outputs (i.e. classes, equals 1 for regression): {context.n_classes}")

        context.output_manager.add_derived_values({
            'n_features': context.n_features,
            'n_testing_ids': sum(len(test_dataset) for test_dataset in test_datasets.values()),
            'n_classes': context.n_classes,
        })

        # Store datasets
        context.train_dataset = train_dataset
        context.val_dataset = val_dataset
        context.test_datasets = test_datasets
        context.baseline_test_datasets = baseline_test_datasets
        context.prediction_dataset = prediction_dataset

        # CLASS WEIGHTS
        context.class_weights = self._get_class_weights(context=context, target_manager=target_manager)

        context.target_manager = target_manager
        return context
