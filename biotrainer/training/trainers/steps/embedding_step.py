import gc
import torch

from pathlib import Path
from typing import Dict, Optional

from junban.pipeline_context import C

from biotrainer_core.data_classes import Protocol, SequenceData

from junban import PipelineStep

from ..pipeline_context import BiotrainerPipelineContext

from ....embedding import get_embedding_service, EmbeddingService, EmbeddingStatsTracker, EmbeddingStats
from ....shared import get_logger

logger = get_logger(__name__)


class EmbeddingStep(PipelineStep[BiotrainerPipelineContext]):

    def _check_entry_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        assert context.input_data is not None, f"Input data cannot be None at the embedding step!"
        return True

    def _check_exit_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        assert context.id2emb is not None and len(context.id2emb) > 0, \
            f"id2emb cannot be None or empty after the embedding step!"
        return True

    def get_start_message(self) -> str:
        return "Embedding..."

    def get_end_message(self) -> str:
        return "Embedding done!"

    def _skip(self, context: BiotrainerPipelineContext) -> bool:
        return context.skip_signal

    @staticmethod
    def _do_embed(context: BiotrainerPipelineContext) -> BiotrainerPipelineContext:
        # Generate embeddings if necessary, otherwise use existing embeddings
        embeddings_file = context.config.get("embeddings_file", None)

        # Track embedding stats
        embedding_stats_tracker = EmbeddingStatsTracker(embedder_name=context.config["embedder_name"])
        embedding_stats = None

        if not embeddings_file:
            # Search for embeddings file at default place if no custom file was provided directly
            embeddings_file = EmbeddingService.get_embeddings_file_path(output_dir=context.config["output_dir"],
                                                                        protocol=context.protocol,
                                                                        embedder_name=context.config["embedder_name"],
                                                                        use_half_precision=context.config.get(
                                                                            "use_half_precision"),
                                                                        )

        if not embeddings_file or not Path(embeddings_file).is_file():
            embedding_service: EmbeddingService = get_embedding_service(
                custom_tokenizer_config=context.config.get("custom_tokenizer_config"),
                embedder_name=context.config["embedder_name"],
                use_half_precision=context.config.get("use_half_precision"),
                device=context.config["device"],
                seed=context.config["seed"],
            )
            embeddings_file = embedding_service.compute_embeddings(
                input_data=context.input_data,
                protocol=context.protocol,
                output_dir=context.config["output_dir"],
                embedding_stats_tracker=embedding_stats_tracker,
            )
            embedding_stats = embedding_stats_tracker.get_stats()
            # Manually clear the memory from costly embedder model
            del embedding_service._embedder
            gc.collect()
        else:
            logger.info(f'Embeddings file was found at {embeddings_file}. Embeddings have not been computed.')

        context.output_manager.update_derived_values(embeddings_file=str(embeddings_file))

        # Mapping from id to embeddings
        ids_to_load = {seq_record.get_hash() for seq_record in context.input_data}
        id2emb = EmbeddingService.load_embeddings(embeddings_file_path=str(embeddings_file),
                                                  ids_to_load=ids_to_load)

        if embedding_stats is None and context.protocol in Protocol.using_per_residue_embeddings():
            assert embedding_stats_tracker.n_tracked == 0, "Embedding stats should be empty!"
            embedding_stats_tracker.track_entire_dataset(id2emb.values())
            embedding_stats = embedding_stats_tracker.get_stats()

        if len(ids_to_load) != len(id2emb):
            raise ValueError(f"Number of embeddings ({len(id2emb)}) != number of input data ({len(ids_to_load)})!")

        assert len(id2emb) > 0, f"No embeddings loaded from embeddings file {embeddings_file}!"

        context.output_manager.update_derived_values(
            embedding_stats=embedding_stats if embedding_stats is not None else None)
        context.id2emb = id2emb
        return context

    @staticmethod
    def _extract_embeddings(context: BiotrainerPipelineContext) -> Optional[Dict[str, torch.Tensor]]:
        input_data = context.input_data
        if isinstance(input_data, list):
            first_value = input_data[0]
            if isinstance(first_value, SequenceData):
                first_embedding = first_value.embedding
                if first_embedding is not None:
                    id2emb = {seq_record.get_id_for_id2emb(): seq_record.get_torch_embedding()
                              for seq_record in input_data}
                    return id2emb
        return None

    def _execute(self, context: BiotrainerPipelineContext) -> BiotrainerPipelineContext:
        maybe_id2emb = self._extract_embeddings(context)
        if maybe_id2emb is not None:
            assert len(maybe_id2emb) == len(
                context.input_data), (f"Number of embeddings ({len(maybe_id2emb)}) "
                                      f"!= number of input data ({len(context.input_data)})!"
                                      f"This should have been caught in the validator earlier.")
            context.id2emb = maybe_id2emb
            logger.info(f"Embeddings were extracted from input data, proceeding...")
            return context
        else:
            return self._do_embed(context)


class FineTuningEmbeddingStep(EmbeddingStep):

    def _execute(self, context: BiotrainerPipelineContext) -> BiotrainerPipelineContext:
        # Calculate / Load embeddings from not-finetuned model once in the beginning for baselines later
        logger.info(f'Finetuning embedder model is activated. Non-finetuned embeddings are calculated once at the '
                    f'beginning to calculate baselines later.')
        context = super()._execute(context)

        embedding_service = get_embedding_service(
            custom_tokenizer_config=context.config.get("custom_tokenizer_config"),
            embedder_name=context.config["embedder_name"],
            use_half_precision=context.config.get("use_half_precision"),
            device=context.config["device"],
            finetuning_config=context.config["finetuning_config"],
            seed=context.config["seed"],
        )
        context.embedding_service = embedding_service
        return context
