import os
import time
import h5py
import torch
import psutil
import numpy as np

from umap import UMAP
from tqdm import tqdm
from pathlib import Path
from numpy import ndarray
from sklearn.manifold import TSNE
from typing import Dict, List, Union, Optional

from .embedder_interfaces import EmbedderInterface

from ..protocols import Protocol
from ..input_files import read_FASTA, BiotrainerSequenceRecord
from ..utilities import get_logger, is_running_in_notebook, calculate_sequence_hash

logger = get_logger(__name__)


class EmbeddingService:
    """
    A service class for computing embeddings using a provided embedder.
    """

    def __init__(self, embedder: EmbedderInterface = None, use_half_precision: bool = False):
        self._embedder = embedder
        self._use_half_precision = use_half_precision

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
        embeddings_file_path = self.get_embeddings_file_path(output_dir=output_dir,
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
        seq_records: List[BiotrainerSequenceRecord]
        if isinstance(input_data, str) or isinstance(input_data, Path):
            seq_records = [seq_record for seq_record in read_FASTA(input_data)]
        elif isinstance(input_data, list):
            if isinstance(input_data[0], BiotrainerSequenceRecord):
                seq_records = input_data
            elif isinstance(input_data[0], str):
                seq_records = [BiotrainerSequenceRecord(seq_id=f"Seq{idx}", seq=seq)
                               for idx, seq in enumerate(input_data)]
            else:
                raise ValueError(f"Non-supported type for compute_embeddings input_data: {type(input_data[0])}")
        elif isinstance(input_data, dict):
            seq_records = [seq_record for seq_record in input_data.values()]
        else:
            raise ValueError(f"Non-supported type for compute_embeddings input_data: {type(input_data)}")

        # Check for not-allowed characters in sequence ids
        if not store_by_hash:
            # / is incompatible because of hierarchical structure of h5
            all_seq_ids_allowed = all(["/" not in seq_record.seq_id for seq_record in seq_records])
            if not all_seq_ids_allowed:
                raise ValueError(f"A sequence id contains the not allowed '/' character, which cannot be stored"
                                 f"in a h5 dataset. Consider changing the id or storing the embeddings by hash.")

        # Sort sequences by length in descending order
        seq_records = list(sorted(seq_records,
                                  key=lambda seq_record: len(seq_record.seq),
                                  reverse=True))

        embeddings_file_path = self._do_embeddings_computation(seq_records=seq_records,
                                                               embeddings_file_path=embeddings_file_path,
                                                               use_reduced_embeddings=use_reduced_embeddings,
                                                               store_by_hash=store_by_hash)

        return str(embeddings_file_path)

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
        embeddings_file_path /= (("reduced_" if use_reduced_embeddings else "")
                                 + f"embeddings_file_{embedder_name}{'_half' if use_half_precision else ''}.h5")
        return embeddings_file_path

    def _do_embeddings_computation(self, seq_records: List[BiotrainerSequenceRecord],
                                   embeddings_file_path: Path,
                                   use_reduced_embeddings: bool,
                                   store_by_hash: bool) -> str:
        """
        Performs the embedding service for the given protein sequences.

        Parameters:
            seq_records (Dict[str, str]): A list of BiotrainerSequenceRecords.
            embeddings_file_path (Path): The path where embeddings will be saved.
            use_reduced_embeddings (bool): Indicates if reduced embeddings should be used.
            store_by_hash (bool): Flag to determine if h5 index should be by hash.

        Returns:
            str: The path to the embeddings file.
        """

        # Get unique seq records by sequence
        seq_records = [seq_record for _, seq_record in
                       {seq_record.seq: seq_record for seq_record in seq_records}.items()
                       ]
        sequences: List[str] = [seq_record.seq for seq_record in seq_records]
        embeddings: List[BiotrainerSequenceRecord] = []  # Seq Records with updated embeddings
        start_time = time.time()

        embedding_iter = self._embedder.embed_many(sequences)
        total_sequences = len(seq_records)

        logger.info("If your dataset contains long reads, it may take more time to process the first few sequences.")

        with tqdm(total=total_sequences, desc="Computing Embeddings", disable=is_running_in_notebook()) as pbar:

            # Load the first sequence and calculate the initial max_embedding_fit
            first_embedding = next(embedding_iter, None)

            if first_embedding is None:
                raise Exception("No embeddings were calculated for the first sequence!")

            pbar.update(1)
            max_embedding_fit = self._max_embedding_fit(first_embedding)
            embedding_dimension = first_embedding.shape[-1]
            embeddings.append(seq_records[0].copy_with_embedding(first_embedding))

            # Load other sequences
            for idx in range(1, total_sequences):
                if max_embedding_fit <= 3 or len(embeddings) % max_embedding_fit == 0 or idx == total_sequences - 1:
                    pbar.desc = "Saving Embeddings"
                    pbar.refresh()
                    embeddings = self._save_and_reset_embeddings(embd_records=embeddings,
                                                                 embeddings_file_path=embeddings_file_path,
                                                                 use_reduced_embeddings=use_reduced_embeddings,
                                                                 store_by_hash=store_by_hash)
                    logger.debug(f"New {max_embedding_fit=}")

                    calculated_embedding = next(embedding_iter, None)
                    embeddings.append(seq_records[idx].copy_with_embedding(calculated_embedding))
                    pbar.desc = "Computing Embeddings"
                    pbar.update(1)
                    pbar.refresh()

                    # Calculate the new max_embedding_fit for the next batch
                    max_embedding_fit = self._max_embedding_fit(calculated_embedding)

                else:
                    calculated_embedding = next(embedding_iter, None)
                    embeddings.append(seq_records[idx].copy_with_embedding(calculated_embedding))
                    pbar.update(1)
                    pbar.refresh()

                    if calculated_embedding is None:
                        raise Exception(f"len(sequence_ids) > len(embedding_iter) or "
                                        f"encountered a None value during embedding calculation!")

        logger.info(f"Embedding dimension: {embedding_dimension}")

        # Save remaining embeddings
        if len(embeddings) > 0:
            pbar.desc = "Saving Embeddings"
            pbar.refresh()
            embeddings = self._save_and_reset_embeddings(embd_records=embeddings,
                                                         embeddings_file_path=embeddings_file_path,
                                                         use_reduced_embeddings=use_reduced_embeddings,
                                                         store_by_hash=store_by_hash)

        end_time = time.time()
        logger.info(f"Time elapsed for computing embeddings: {end_time - start_time:.2f}[s]")

        return str(embeddings_file_path)

    @staticmethod
    def _max_embedding_fit(embedding: ndarray) -> int:
        """
        Calculates the maximum number of embeddings that can fit in available memory.

        This function estimates the maximum number of embeddings that can be stored in 
        the available system memory without exceeding it. The calculation includes a 
        safety factor to prevent exhausting memory.

        Parameters:
            embedding (ndarray): An embedding array, representing the data structure 
                                 whose memory footprint is being considered.

        Returns:
            int: The maximum number of embeddings that can fit in memory.

        Notes:
            - The number 18 was determined experimentally as a factor correlating the 
              embedding size to the memory usage, indicating that each unit of 
              embedding size corresponds to approximately 18 bytes of memory.
            - The multiplier 0.75 is a safety margin to ensure that the memory usage 
              stays within 75% of the available system memory, reducing the risk of 
              running out of RAM during operations.
        """
        max_embedding_fit = int(0.75 * (psutil.virtual_memory().available / (embedding.size * 18)))
        max_embedding_fit = 1 if max_embedding_fit == 0 else max_embedding_fit
        return max_embedding_fit

    def _save_and_reset_embeddings(self, embd_records: List[BiotrainerSequenceRecord],
                                   embeddings_file_path: Path,
                                   use_reduced_embeddings: bool,
                                   store_by_hash: bool) -> List[BiotrainerSequenceRecord]:
        """
        Save the embeddings and reset the list.

        Parameters:
            embd_records (List[BiotrainerSequenceRecord]): List of seq records with embeddings to be saved.
            embeddings_file_path (Path): The path where embeddings are saved.
            use_reduced_embeddings (bool): Flag to determine if embeddings should be reduced.
            store_by_hash (bool): Flag to determine if h5 index should be by hash.
        Returns:
           Deleted and emptied seq records list
        """
        if use_reduced_embeddings:
            embd_records = self._reduce_embeddings(embd_records, self._embedder)
        self._save_embeddings(embd_records=embd_records, embeddings_file_path=embeddings_file_path,
                              store_by_hash=store_by_hash)
        del embd_records
        return []

    @staticmethod
    def embeddings_dimensionality_reduction(
            embeddings: Dict[str, torch.tensor],
            dimension_reduction_method: str,
            n_reduced_components: int) -> Dict[str, torch.tensor]:
        """Reduces the dimension of per-protein embeddings using one of the
        dimensionality reduction methods

        Args:
            embeddings (Dict[str, torch.tensor]): Dictionary of embeddings.
            dimension_reduction_method (str): The method used to reduce 
            the dimensionality of embeddings. Options are 'umap' or 'tsne'.
            n_reduced_components (int): The target number of dimensions for 
            the reduced embeddings.

        Returns:
            Dict[str, torch.tensor]: Dictionary of embeddings with reduced dimensions.
        """
        sorted_keys = sorted(list(embeddings.keys()))
        all_embeddings = torch.stack([embeddings[k] for k in sorted_keys], dim=0)
        max_dim_dict = {
            "umap": all_embeddings.shape[0] - 2,
            "tsne": all_embeddings.shape[0] - 1
        }
        n_reduced_components = min([
            n_reduced_components,
            max_dim_dict[dimension_reduction_method],
            all_embeddings.shape[1]])
        dimension_reduction_method_dict = {
            "umap": UMAP(n_components=n_reduced_components),
            "tsne": TSNE(
                n_components=n_reduced_components,
                perplexity=min(30, n_reduced_components))
        }
        logger.info(f"Starting embeddings dimensionality reduction via method {dimension_reduction_method}")
        embeddings_reduced_dimensions = dimension_reduction_method_dict[
            dimension_reduction_method].fit_transform(all_embeddings)
        logger.info(f"Finished embeddings dimensionality reduction!")
        return {sorted_keys[i]: torch.tensor(embeddings_reduced_dimensions[i]) for i in range(len(sorted_keys))}

    @staticmethod
    def _reduce_embeddings(embd_records: List[BiotrainerSequenceRecord], embedder) -> List[BiotrainerSequenceRecord]:
        """
        Reduces the per-residue embeddings to per-protein embeddings.

        Parameters:
            embd_records (List[BiotrainerSequenceRecord]): Dictionary of seq records with embeddings.
            embedder: The embedder used for reducing embeddings.

        Returns:
            out (List[BiotrainerSequenceRecord]): Dictionary of seq records with reduced embeddings.
        """
        return [seq_record.copy_with_embedding(embedder.reduce_per_protein(seq_record.embedding))
                for seq_record in embd_records]

    @staticmethod
    def _save_embeddings(embd_records: List[BiotrainerSequenceRecord], embeddings_file_path: Path, store_by_hash: bool):
        """
        Saves the embeddings to a file.

        Args:
            embd_records (List[BiotrainerSequenceRecord]): List of seq records with embeddings to be saved.
            embeddings_file_path (Path): The path where embeddings are saved.

        Returns:
            out (int): The updated save ID.
        """
        with h5py.File(embeddings_file_path, "a") as embeddings_file:
            for seq_record in embd_records:
                h5_index = seq_record.seq_id
                if store_by_hash:
                    h5_index = seq_record.get_hash()
                embeddings_file.create_dataset(h5_index, data=seq_record.embedding, compression="gzip", chunks=True)
                embeddings_file[h5_index].attrs[
                    "original_id"] = seq_record.seq_id  # Follows biotrainer & bio_embeddings standard

    def compute_embeddings_from_list(self, protein_sequences: List[str], protocol: Protocol) -> List:
        """
        Compute embeddings with the provided embedder directly from a list of sequences.

        Parameters:
            protein_sequences (List[str]): List of protein sequences as strings.
            protocol (Protocol): Protocol for the embeddings. Determines if the embeddings should be reduced to per-protein.

        Returns:
            out (List): List of computed embeddings.
        """
        use_reduced_embeddings = protocol in Protocol.using_per_sequence_embeddings()

        embeddings = list(tqdm(self._embedder.embed_many(protein_sequences), total=len(protein_sequences),
                               disable=is_running_in_notebook()))

        if use_reduced_embeddings:
            embeddings = [self._embedder.reduce_per_protein(embedding) for embedding in embeddings]
        return embeddings

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
        id2emb = {idx: torch.tensor(np.array(embedding)) for (idx, embedding) in embeddings_file.items()}

        # Logging
        logger.info(f"Read {len(id2emb)} entries.")
        logger.info(f"Time elapsed for reading embeddings: {(time.perf_counter() - start):.1f}[s]")

        return id2emb
