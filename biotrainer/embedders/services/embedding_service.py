from __future__ import annotations

import os
import time
import h5py
import torch

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Union, Optional, Generator, Tuple, Any

from ..interfaces import EmbedderInterface

from ...protocols import Protocol
from ...utilities import get_logger, is_running_in_notebook
from ...input_files import read_FASTA, BiotrainerSequenceRecord

logger = get_logger(__name__)


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
                           input_data: Union[str, Path, List[str],
                           List[BiotrainerSequenceRecord], Dict[str, BiotrainerSequenceRecord]],
                           output_dir: Path,
                           protocol: Protocol,
                           force_output_dir: bool = False,
                           force_recomputing: bool = False,
                           store_by_hash: bool = True) -> str:
        """
        Compute embeddings with the provided embedder from a sequence file or a dictionary of sequences.

        Parameters:
            input_data (Union[str, Dict[str, str]]): Path to the sequence file or a dictionary of protein sequences.
            output_dir (Path): Output directory to store the computed embeddings.
            protocol (Protocol): Protocol for the embeddings. Determines if the embeddings should be reduced to per-protein.
            force_output_dir (bool): If True, the given output directory is directly used to store the embeddings file.
            force_recomputing (bool): If True, the embedding file is re-computed, even if it already exists.
            store_by_hash (bool): If True, sequence hashes are used as indices for the h5 result file.

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

        # Avoid re-computation if file already exists
        if not force_recomputing and embeddings_file_path.is_file():
            logger.info(f"Using existing embeddings file at {embeddings_file_path}")
            return str(embeddings_file_path)

        logger.info(f"Computing embeddings to: {str(embeddings_file_path)}")

        # Process input data
        seq_records = self._process_input_data(input_data)

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

        # Open the h5 file once and write embeddings as they're generated
        with h5py.File(embeddings_file_path, "a") as embeddings_file:
            for seq_record, embedding in tqdm(
                    self._embeddings_generator(seq_records, use_reduced_embeddings),
                    total=len(seq_records),
                    desc="Computing Embeddings",
                    disable=is_running_in_notebook()
            ):
                self.store_embedding(embeddings_file_handle=embeddings_file,
                                     seq_record=seq_record,
                                     embedding=embedding,
                                     store_by_hash=store_by_hash)

        end_time = time.time()
        logger.info(f"Time elapsed for computing embeddings: {end_time - start_time:.2f}[s]")

        return str(embeddings_file_path)

    @staticmethod
    def store_embedding(embeddings_file_handle, seq_record, embedding, store_by_hash: bool = True):
        h5_index = seq_record.get_hash() if store_by_hash else seq_record.seq_id
        embeddings_file_handle.create_dataset(h5_index, data=embedding.cpu().numpy(), compression="gzip", chunks=True)
        embeddings_file_handle[h5_index].attrs["original_id"] = seq_record.seq_id

    def generate_embeddings(self,
                            input_data: Union[str, Path, List[str], List[BiotrainerSequenceRecord], Dict[
                                str, BiotrainerSequenceRecord]],
                            reduce: bool) -> Generator[Tuple[BiotrainerSequenceRecord, torch.tensor], None, None]:
        """
        Generator function that yields embeddings as they are computed.

        Parameters:
            input_data: Input sequences in various formats
            reduce: If True, embeddings will be reduced to per-sequence embeddings.

        Yields:
            Tuple[BiotrainerSequenceRecord, torch.tensor]: Tuple of (BiotrainerSequenceRecord, embedding)
        """

        # Process input data
        seq_records = self._process_input_data(input_data)

        # Sort sequences by length in descending order
        seq_records = list(sorted(seq_records,
                                  key=lambda seq_record: len(seq_record.seq),
                                  reverse=True))

        # Generate embeddings
        yield from self._embeddings_generator(seq_records, reduce)

    @staticmethod
    def _process_input_data(input_data) -> List[BiotrainerSequenceRecord]:
        """
        Process various input formats into a list of BiotrainerSequenceRecord
        """
        if isinstance(input_data, (str, Path)):
            return [seq_record for seq_record in read_FASTA(input_data)]
        elif isinstance(input_data, list):
            if isinstance(input_data[0], BiotrainerSequenceRecord):
                return input_data
            elif isinstance(input_data[0], str):
                return [BiotrainerSequenceRecord(seq_id=f"Seq{idx}", seq=seq)
                        for idx, seq in enumerate(input_data)]
            else:
                raise ValueError(f"Non-supported type for input_data: {type(input_data[0])}")
        elif isinstance(input_data, dict):
            return [seq_record for seq_record in input_data.values()]
        else:
            raise ValueError(f"Non-supported type for input_data: {type(input_data)}")

    def _embeddings_generator(self,
                              seq_records: List[BiotrainerSequenceRecord],
                              use_reduced_embeddings: bool) \
            -> Generator[Tuple[BiotrainerSequenceRecord, torch.tensor], None, None]:
        """
        Core embedding computation logic that can be used by both save and generate methods

        Parameters:
            seq_records: List of sequence records to process
            use_reduced_embeddings: Whether to reduce embeddings to per-protein

        Yields:
            Tuple[BiotrainerSequenceRecord, torch.tensor]: Tuple of (BiotrainerSequenceRecord, embedding)
        """
        sequences = [seq_record.seq for seq_record in seq_records]
        embedding_iter = self._embedder.embed_many(sequences)

        for seq_record, embedding in zip(seq_records, embedding_iter):
            if embedding is None:
                raise Exception("Encountered None value during embedding calculation!")

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
            embeddings_file_path = output_dir / protocol.name
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
            embeddings: Dict[str, torch.tensor],
            dimension_reduction_method: str,
            n_reduced_components: int,
            fitted_transform: Optional[Any] = None) -> Tuple[Dict[str, torch.tensor], Any]:
        """Reduces the dimension of per-protein embeddings using one of the
        dimensionality reduction methods

        Args:
            embeddings (Dict[str, torch.tensor]): Dictionary of embeddings.
            dimension_reduction_method (str): The method used to reduce
            the dimensionality of embeddings. Options are 'umap' or 'pca'.
            n_reduced_components (int): The target number of dimensions for
            the reduced embeddings.
            fitted_transform (Any): Existing fitted transform object. If provided, the given transform is applied to
                the embeddings instead of calculating a new transform.

        Returns:
            Dict[str, torch.tensor]: Dictionary of embeddings with reduced dimensions.
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
    def load_embeddings(embeddings_file_path: str) -> Dict[str, torch.tensor]:
        """
        Loads precomputed embeddings from a file.

        Parameters:
            embeddings_file_path (str): Path to the embeddings file.

        Returns:
            out (Dict[str, torch.tensor]): Dictionary mapping sequence hashes to embeddings.
        """
        # Load computed embeddings in .h5 file format
        logger.info(f"Loading embeddings from: {embeddings_file_path}")
        start = time.perf_counter()

        # Old version see:
        # https://stackoverflow.com/questions/48385256/optimal-hdf5-dataset-chunk-shape-for-reading-rows/48405220#48405220
        embeddings_file = h5py.File(embeddings_file_path, 'r')

        # Sequence hash from embeddings file -> Embedding
        id2emb = {idx: torch.tensor(embedding).float() for (idx, embedding) in embeddings_file.items()}

        # Logging
        logger.info(f"Read {len(id2emb)} entries.")
        logger.info(f"Time elapsed for reading embeddings: {(time.perf_counter() - start):.1f}[s]")

        return id2emb
