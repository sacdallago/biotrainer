import gc

from pathlib import Path

from ..pipeline import PipelineContext, PipelineStep
from ..pipeline.pipeline_step import PipelineStepType

from ...utilities import get_logger
from ...embedders import get_embedding_service, EmbeddingService

logger = get_logger(__name__)


class EmbeddingStep(PipelineStep):

    def get_step_type(self) -> PipelineStepType:
        return PipelineStepType.EMBEDDING

    def process(self, context: PipelineContext) -> PipelineContext:
        # Generate embeddings if necessary, otherwise use existing embeddings
        embeddings_file = context.config.get("embeddings_file", None)

        if not embeddings_file:
            # Search for embeddings file at default place if no custom file was provided directly
            embeddings_file = EmbeddingService.get_embeddings_file_path(output_dir=context.config["output_dir"],
                                                                        protocol=context.config["protocol"],
                                                                        embedder_name=context.config["embedder_name"],
                                                                        use_half_precision=context.config.get(
                                                                            "use_half_precision"),
                                                                        )

        if not embeddings_file or not Path(embeddings_file).is_file():
            embedding_service: EmbeddingService = get_embedding_service(
                custom_tokenizer_config=context.config.get("custom_tokenizer_config"),
                embedder_name=context.config["embedder_name"],
                use_half_precision=context.config.get("use_half_precision"),
                device=context.config["device"]
            )
            embeddings_file = embedding_service.compute_embeddings(
                input_data=context.input_data,
                protocol=context.config["protocol"], output_dir=context.config["output_dir"]
            )

            # Manually clear the memory from costly embedder model
            del embedding_service._embedder
            gc.collect()
        else:
            logger.info(f'Embeddings file was found at {embeddings_file}. Embeddings have not been computed.')

        context.output_manager.add_derived_values({'embeddings_file': str(embeddings_file)})

        # Mapping from id to embeddings
        id2emb = EmbeddingService.load_embeddings(embeddings_file_path=str(embeddings_file))

        context.id2emb = id2emb
        return context


class FineTuningEmbeddingStep(EmbeddingStep):
    def process(self, context: PipelineContext) -> PipelineContext:
        # Calculate / Load embeddings from not-finetuned model once in the beginning for baselines later
        logger.info(f'Finetuning embedder model is activated. Non-finetuned embeddings are calculated once at the '
                    f'beginning to calculate baselines later.')
        context = super().process(context)

        embedding_service = get_embedding_service(
            custom_tokenizer_config=context.config.get("custom_tokenizer_config"),
            embedder_name=context.config["embedder_name"],
            use_half_precision=context.config.get("use_half_precision"),
            device=context.config["device"],
            finetuning_config=context.config["finetuning_config"]
        )

        context.embedding_service = embedding_service
        return context
