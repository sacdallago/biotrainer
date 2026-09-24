from __future__ import annotations

import os
import time
import h5py
import torch
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from biotrainer_core.input_files import read_FASTA
from biotrainer_core.data_classes import SequenceData, Protocol
from biotrainer_core.h5_files import read_id2emb, store_embedding_to_handle
from typing import Dict, List, Union, Optional, Generator, Tuple, Any, Set

from ..stats import EmbeddingStatsTracker
from ..interfaces import EmbedderInterface

from ...shared import get_logger, is_running_in_notebook

logger = get_logger(__name__)


def _io_worker_process(queue: mp.Queue, file_path: Path, store_by_hash: bool):
    """ Parallel process to store embeddings to h5 file while computing embeddings """
    import h5py  # local import to ensure availability in spawned process

    with h5py.File(file_path, "a") as embeddings_file:
        # Iterate over items from the queue until the sentinel None is received
        for emb_record in iter(queue.get, None):
            EmbeddingService.store_embedding(
                embeddings_file, emb_record, store_by_hash
            )


class EmbeddingService:
    """
    A service class for computing embeddings using a provided embedder.
    """

    def __init__(self, embedder: EmbedderInterface = None, use_half_precision: bool = False):
        self._embedder = embedder
        self._use_half_precision = use_half_precision

    def save_embedder(self, output_dir: Path):
        """Save fine-tuned model after training"""
        logger.warning("Trying to save non-finetuned model!")

    def add_finetuned_adapter(self, adapter_path: Path) -> EmbeddingService:
        """Add finetuned adapter after training to embedder model. Used for inference."""
        from peft import PeftModel
        # Apply LoRA adapters to the embedder model
        model = self._embedder._model
        if model is None:
            raise ValueError(f"{self._embedder.name} does not provide a model to add the finetuning adapter!")

        model = PeftModel.from_pretrained(
            self._embedder._model,
            adapter_path
        )

        self._embedder._model = model.eval()

        print(f"Added fine-tuned adapter to {self._embedder.name}: {adapter_path}")
        return self

    def compute_embeddings(self,
                           input_data: Union[str, Path, List[str], List[SequenceData], Dict[str, SequenceData]],
                           output_dir: Path,
                           protocol: Protocol,
                           force_output_dir: bool = False,
                           force_recomputing: bool = False,
                           store_by_hash: bool = True,
                           embedding_stats_tracker: EmbeddingStatsTracker = None) -> str:
        """
        Compute embeddings with the provided embedder from a sequence file or a dictionary of sequences.

        Parameters:
            input_data: Path to the sequence file or an iterable of protein sequences.
            output_dir (Path): Output directory to store the computed embeddings.
            protocol (Protocol): Protocol for the embeddings. Determines if the embeddings should be reduced to per-protein.
            force_output_dir (bool): If True, the given output directory is directly used to store the embeddings file.
            force_recomputing (bool): If True, the embedding file is re-computed, even if it already exists.
            store_by_hash (bool): If True, sequence hashes are used as indices for the h5 result file.
            embedding_stats_tracker (EmbeddingStatsTracker): Optional tracker for tracking embedding statistics.
        Returns:
            str: Path to the generated output h5 embeddings file.
        """
        use_reduced_embeddings = protocol in Protocol.using_per_sequence_embeddings()
        embeddings_file_path = self.get_embeddings_file_path(
            output_dir=output_dir,
            protocol=protocol,
            embedder_name=self._embedder.name,
            use_half_precision=self._use_half_precision,
            force_output_dir=force_output_dir
        )

        # Avoid re-computation if file already exists and all sequences are present
        if force_recomputing and embeddings_file_path.is_file():
            embeddings_file_path.unlink()

        # Process input data
        seq_records = self._process_input_data(input_data)

        if not force_recomputing and embeddings_file_path.is_file():
            with h5py.File(embeddings_file_path, "r") as embeddings_file:
                existing_keys = set([k for k in tqdm(embeddings_file.keys(), desc=f"Looking up existing embeddings..")])
            if store_by_hash:
                seq_records_not_existing_yet = [seq_record for seq_record in seq_records
                                                if seq_record.get_hash() not in existing_keys]
            else:
                seq_records_not_existing_yet = [seq_record for seq_record in seq_records
                                                if seq_record.seq_id not in existing_keys]

            if len(seq_records_not_existing_yet) == 0:
                logger.info(f"Using existing embeddings file at {embeddings_file_path}")
                return str(embeddings_file_path)

            n_already_existing = len(seq_records) - len(seq_records_not_existing_yet)
            logger.info(f"Using {n_already_existing} existing embeddings from file, "
                        f"embedding {len(seq_records_not_existing_yet)} new sequences..")
            seq_records = seq_records_not_existing_yet

        logger.info(f"Computing embeddings to: {str(embeddings_file_path)}")

        # Check for not-allowed characters in sequence ids
        if not store_by_hash:
            all_seq_ids_allowed = all(["/" not in seq_record.seq_id for seq_record in seq_records])
            if not all_seq_ids_allowed:
                raise ValueError("A sequence id contains the not allowed '/' character")

        # Sort sequences by length in descending order
        seq_records = list(sorted(seq_records,
                                  key=lambda seq_record: len(seq_record.seq),
                                  reverse=True))

        start_time = time.time()

        self._compute_embeddings_parallel(
            seq_records=seq_records,
            embeddings_file_path=embeddings_file_path,
            use_reduced_embeddings=use_reduced_embeddings,
            store_by_hash=store_by_hash,
            embedding_stats_tracker=embedding_stats_tracker,
        )

        end_time = time.time()
        logger.info(f"Time elapsed for computing embeddings: {end_time - start_time:.2f}[s]")

        return str(embeddings_file_path)

    def _compute_embeddings_parallel(self,
                                     seq_records: List[SequenceData],
                                     embeddings_file_path: Path,
                                     use_reduced_embeddings: bool,
                                     store_by_hash: bool,
                                     embedding_stats_tracker: EmbeddingStatsTracker = None):
        """
        Use separate process for I/O and run embedding on main process/GPU.
        """
        ctx = mp.get_context('spawn')
        embedding_queue = ctx.Queue(maxsize=30)

        # Start single I/O worker
        io_process = mp.Process(
            target=_io_worker_process,
            args=(embedding_queue, embeddings_file_path, store_by_hash)
        )
        io_process.start()

        try:
            for seq_record, embedding in tqdm(
                    self._embeddings_generator(seq_records, use_reduced_embeddings, embedding_stats_tracker),
                    total=len(seq_records),
                    desc="Computing Embeddings",
                    disable=is_running_in_notebook()
            ):
                # Convert to numpy and move to CPU before putting in queue
                embedding_np = embedding.cpu().numpy()

                emb_record = seq_record.copy_with_embedding(embedding=embedding_np)

                embedding_queue.put(emb_record)
            # Signal worker to stop after embeddings are computed
            embedding_queue.put(None)
        except Exception as e:
            embedding_queue.put(None) # Stop on Exception
            raise e
        finally:
            io_process.join()

    @staticmethod
    def store_embedding(embeddings_file_handle,
                        emb_record: SequenceData,
                        store_by_hash: bool = True) -> bool:
        h5_index = emb_record.get_hash() if store_by_hash else emb_record.seq_id

        # Handle both torch tensors and numpy arrays
        embedding = emb_record.embedding
        if isinstance(embedding, torch.Tensor):
            embedding_data = embedding.cpu().numpy()
        else:
            embedding_data = embedding

        return store_embedding_to_handle(embeddings_file_handle, h5_index, emb_record.seq_id, embedding_data)

    def generate_embeddings(self,
                            input_data: Union[str, Path, List[str], List[SequenceData], Dict[str, SequenceData]],
                            reduce: bool) -> Generator[SequenceData, None, None]:
        """
        Generator function that yields embeddings as they are computed.

        Parameters:
            input_data: Input sequences in various formats
            reduce: If True, embeddings will be reduced to per-sequence embeddings.

        Yields:
            SequenceData: SequenceData object with the computed embedding (as torch.tensor).
        """

        # Process input data
        seq_records = self._process_input_data(input_data)

        # Sort sequences by length in descending order
        seq_records = list(sorted(seq_records,
                                  key=lambda seq_record: len(seq_record.seq),
                                  reverse=True))

        # Generate embeddings
        for seq_record, embedding in self._embeddings_generator(seq_records, reduce):
            emb_record = seq_record.copy_with_embedding(embedding=embedding)
            yield emb_record

    @staticmethod
    def _process_input_data(input_data) -> List[SequenceData]:
        """
        Process various input formats into a list of SequenceData
        """
        if isinstance(input_data, (str, Path)):
            return [seq_record for seq_record in read_FASTA(input_data)]
        elif isinstance(input_data, list):
            if len(input_data) == 0:
                raise ValueError(
                    "Input data is an empty list. Please provide a non-empty list of sequences."
                )
            if isinstance(input_data[0], SequenceData):
                return input_data
            elif isinstance(input_data[0], str):
                return [SequenceData(seq_id=f"Seq{idx}", seq=seq)
                        for idx, seq in enumerate(input_data)]
            else:
                raise ValueError(f"Non-supported type for input_data: {type(input_data[0])}")
        elif isinstance(input_data, dict):
            return [seq_record for seq_record in input_data.values()]
        else:
            raise ValueError(f"Non-supported type for input_data: {type(input_data)}")

    def _embeddings_generator(self,
                              seq_records: List[SequenceData],
                              use_reduced_embeddings: bool,
                              embedding_stats_tracker: Optional[EmbeddingStatsTracker] = None) \
            -> Generator[Tuple[SequenceData, torch.Tensor], None, None]:
        """
        Core embedding computation logic that can be used by both save and generate methods

        Parameters:
            seq_records: List of sequence records to process
            use_reduced_embeddings: Whether to reduce embeddings to per-protein

        Yields:
            Tuple[BiotrainerSequenceRecord, torch.Tensor]: Tuple of (BiotrainerSequenceRecord, embedding)
        """
        sequences = [seq_record.seq for seq_record in seq_records]
        embedding_iter = self._embedder.embed_many(sequences)

        for seq_record, embedding in zip(seq_records, embedding_iter):
            if embedding is None:
                raise Exception("Encountered None value during embedding calculation!")

            if embedding_stats_tracker is not None:
                embedding_stats_tracker.track(embedding)

            if use_reduced_embeddings:
                # TODO Batching might improve speed here
                embedding = self._embedder.reduce_per_protein(embedding)

            yield seq_record, embedding

    @staticmethod
    def get_embeddings_file_path(output_dir: Path,
                                 protocol: Protocol,
                                 embedder_name: str,
                                 use_half_precision: bool,
                                 force_output_dir: Optional[bool] = False) -> Path:
        use_reduced_embeddings = protocol in Protocol.using_per_sequence_embeddings()
        embedder_name = embedder_name.split("/")[-1]

        if force_output_dir:
            embeddings_file_path = output_dir
        else:
            # Create protocol path to embeddings
            embeddings_file_path = output_dir / "embeddings"
            if not os.path.isdir(embeddings_file_path):
                os.mkdir(embeddings_file_path)

            # Add embedder name subdirectory
            embeddings_file_path /= embedder_name
            if not os.path.isdir(embeddings_file_path):
                os.mkdir(embeddings_file_path)

        # Append file name to output path
        embeddings_file_path /= EmbeddingService.get_embeddings_file_name(embedder_name,
                                                                          use_half_precision,
                                                                          use_reduced_embeddings)
        return embeddings_file_path

    @staticmethod
    def get_embeddings_file_name(embedder_name: str,
                                 use_half_precision: bool,
                                 use_reduced_embeddings: bool):
        embedder_name = embedder_name.split("/")[-1]
        return (("reduced_" if use_reduced_embeddings else "")
                + f"embeddings_file_{embedder_name}{'_half' if use_half_precision else ''}.h5")

    @staticmethod
    def embeddings_dimensionality_reduction(
            embeddings: Dict[str, torch.Tensor],
            dimension_reduction_method: str,
            n_reduced_components: int,
            fitted_transform: Optional[Any] = None) -> Tuple[Dict[str, torch.Tensor], Any]:
        """Reduces the dimension of per-protein embeddings using one of the
        dimensionality reduction methods

        Args:
            embeddings (Dict[str, torch.Tensor]): Dictionary of embeddings.
            dimension_reduction_method (str): The method used to reduce
            the dimensionality of embeddings. Options are 'umap' or 'pca'.
            n_reduced_components (int): The target number of dimensions for
            the reduced embeddings.
            fitted_transform (Any): Existing fitted transform object. If provided, the given transform is applied to
                the embeddings instead of calculating a new transform.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of embeddings with reduced dimensions.
        """
        from umap import UMAP
        from sklearn.decomposition import PCA

        sorted_keys = sorted(list(embeddings.keys()))
        first_embedding = embeddings[sorted_keys[0]]
        dtype_embedding = first_embedding.dtype
        all_embeddings = torch.stack([embeddings[k] for k in sorted_keys], dim=0)
        if fitted_transform is None:
            max_dim_dict = {
                "umap": all_embeddings.shape[0] - 2,
                "pca": min(all_embeddings.shape[0], all_embeddings.shape[1])
            }
            n_reduced_components = min([
                n_reduced_components,
                max_dim_dict[dimension_reduction_method],
                all_embeddings.shape[1]]
            )
            dimension_reduction_method_dict = {
                "umap": UMAP(n_components=n_reduced_components),
                "pca": PCA(n_components=n_reduced_components)
            }
            logger.info(f"Starting embeddings dimensionality reduction via method {dimension_reduction_method}")
            fitted_transform = dimension_reduction_method_dict[dimension_reduction_method]
            embeddings_reduced_dimensions = fitted_transform.fit_transform(all_embeddings)
        else:
            embeddings_reduced_dimensions = fitted_transform.transform(all_embeddings)

        reduced_embeddings = {sorted_keys[i]: torch.tensor(embeddings_reduced_dimensions[i], dtype=dtype_embedding)
                              for i in range(len(sorted_keys))}

        return reduced_embeddings, fitted_transform

    @staticmethod
    def load_embeddings(embeddings_file_path: str, ids_to_load: Optional[Set[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Loads precomputed embeddings from a file.

        Parameters:
            embeddings_file_path (str): Path to the embeddings file.
            ids_to_load (Optional[List[str]]): List of sequence hashes to load. If None, all sequences are loaded.
        Returns:
            out (Dict[str, torch.Tensor]): Dictionary mapping sequence hashes to embeddings.
        """
        # Load computed embeddings in .h5 file format
        logger.info(f"Loading embeddings from: {embeddings_file_path}")
        start = time.perf_counter()

        # Sequence hash from embeddings file -> Embedding
        id2emb = read_id2emb(embeddings_file_path, ids_to_load)
        id2emb = {idx: torch.tensor(embedding).float() for (idx, embedding) in id2emb.items()}

        # Logging
        logger.info(f"Read {len(id2emb)} entries.")
        logger.info(f"Time elapsed for reading embeddings: {(time.perf_counter() - start):.1f}[s]")

        return id2emb
