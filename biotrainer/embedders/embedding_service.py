import os
import gc
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
from typing import Dict, Tuple, Any, List, Union, Optional

from .embedder_interfaces import EmbedderInterface

from ..protocols import Protocol
from ..utilities import read_FASTA, get_logger, is_running_in_notebook

logger = get_logger(__name__)


class EmbeddingService:
    """
    A service class for computing embeddings using a provided embedder.
    """

    def __init__(self, embedder: EmbedderInterface = None, use_half_precision: bool = False):
        self._embedder = embedder
        self._use_half_precision = use_half_precision

    def compute_embeddings(self,
                           input_data: Union[str, Path, Dict[str, str]],
                           output_dir: Path,
                           protocol: Protocol,
                           force_output_dir: bool = False,
                           force_recomputing: bool = False) -> str:
        """
        Compute embeddings with the provided embedder from a sequence file or a dictionary of sequences.

        Parameters:
            input_data (Union[str, Dict[str, str]]): Path to the sequence file or a dictionary of protein sequences.
            output_dir (Path): Output directory to store the computed embeddings.
            protocol (Protocol): Protocol for the embeddings. Determines if the embeddings should be reduced to per-protein.
            force_output_dir (bool): If True, the given output directory is directly used to store the embeddings file.
            force_recomputing (bool): If True, the embedding file is re-computed, even if it already exists.

        Returns:
            str: Path to the generated output embeddings file.
        """
        use_reduced_embeddings = protocol in Protocol.using_per_sequence_embeddings()
        embeddings_file_path = self._get_embeddings_file_path(output_dir, protocol, force_output_dir)

        # Avoid re-computation if file already exists
        if not force_recomputing and embeddings_file_path.is_file():
            logger.info(f"Using existing embeddings file at {embeddings_file_path}")
            return str(embeddings_file_path)

        logger.info(f"Computing embeddings to: {str(embeddings_file_path)}")

        # Process input data
        if isinstance(input_data, str):
            protein_sequences = {seq.id: str(seq.seq) for seq in read_FASTA(input_data)}
        elif isinstance(input_data, Path):
            protein_sequences = {seq.id: str(seq.seq) for seq in read_FASTA(str(input_data))}
        elif isinstance(input_data, dict):
            protein_sequences = input_data
        else:
            raise ValueError("input_data must be either a file path or a dictionary of sequences")

        # Sort sequences by length in descending order
        protein_sequences = dict(sorted(protein_sequences.items(),
                                        key=lambda item: len(item[1]),
                                        reverse=True))

        embeddings_file_path = self._do_embeddings_computation(protein_sequences,
                                                               embeddings_file_path,
                                                               use_reduced_embeddings)

        return str(embeddings_file_path)

    def _get_embeddings_file_path(self, output_dir: Path, protocol: Protocol,
                                  force_output_dir: Optional[bool] = False) -> Path:
        use_reduced_embeddings = protocol in Protocol.using_per_sequence_embeddings()
        embedder_name = self._embedder.name.split("/")[-1]

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
                                 + f"embeddings_file_{embedder_name}{'_half' if self._use_half_precision else ''}.h5")
        return embeddings_file_path

    def _do_embeddings_computation(self, protein_sequences: Dict[str, str], embeddings_file_path: Path,
                                   use_reduced_embeddings: bool) -> str:
        """
        Performs the embedding service for the given protein sequences.

        Parameters:
            protein_sequences (Dict[str, str]): A dictionary mapping sequence IDs to protein sequences.
            embeddings_file_path (Path): The path where embeddings will be saved.
            use_reduced_embeddings (bool): Indicates if reduced embeddings should be used.

        Returns:
            str: The path to the embeddings file.
        """
        sequence_ids = list(protein_sequences.keys())
        embeddings = {}
        idx: int = 0
        last_save_id: int = 0
        start_time = time.time()

        embedding_iter = self._embedder.embed_many(protein_sequences.values())
        total_sequences = len(protein_sequences.values())

        logger.info("If your dataset contains long reads, it may take more time to process the first few sequences.")

        with tqdm(total=total_sequences, desc="Computing Embeddings", disable=is_running_in_notebook()) as pbar:

            # Load the first sequence and calculate the initial max_embedding_fit
            embeddings[sequence_ids[0]] = next(embedding_iter, None)
            pbar.update(1)

            max_embedding_fit = self._max_embedding_fit(embeddings[sequence_ids[0]])

            if embeddings[sequence_ids[0]] is None:
                logger.info(f"No embeddings found.")
                return str(embeddings_file_path)

            embedding_dimension = embeddings[sequence_ids[0]].shape[-1]

            # Load other sequences
            for idx in range(1, total_sequences):
                if max_embedding_fit <= 3 or len(embeddings) % max_embedding_fit == 0 or idx == total_sequences - 1:
                    pbar.desc = "Saving Embeddings"
                    last_save_id, embeddings = self._save_and_reset_embeddings(embeddings, last_save_id,
                                                                               embeddings_file_path,
                                                                               use_reduced_embeddings)
                    logger.debug(f"New {max_embedding_fit=}")

                    embeddings[sequence_ids[idx]] = next(embedding_iter, None)
                    pbar.desc = "Computing Embeddings"
                    pbar.update(1)

                    # Calculate the new max_embedding_fit for the next batch
                    max_embedding_fit = self._max_embedding_fit(embeddings[sequence_ids[idx]])

                else:
                    embeddings[sequence_ids[idx]] = next(embedding_iter, None)
                    pbar.update(1)

                    if embeddings[sequence_ids[idx]] is None:
                        logger.debug(
                            f"len(sequence_ids) > len(embedding_iter) or found a None value in the embedding_iter")
                        del embeddings[sequence_ids[idx]]
                        return str(embeddings_file_path)

        logger.info(f"Embedding dimension: {embedding_dimension}")

        # Save remaining embeddings
        if len(embeddings) > 0:
            last_save_id, embeddings = self._save_and_reset_embeddings(embeddings, last_save_id,
                                                                       embeddings_file_path, use_reduced_embeddings)

        end_time = time.time()
        logger.info(f"Time elapsed for computing embeddings: {end_time - start_time:.2f}[s]")

        del embeddings
        del self._embedder
        gc.collect()

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

    def _save_and_reset_embeddings(self, embeddings: Dict[str, ndarray], last_save_id: int,
                                   embeddings_file_path: Path, use_reduced_embeddings: bool) -> Tuple[
        int, Dict[str, ndarray]]:
        """
        Save the embeddings and reset the dictionary.

        Parameters:
            embeddings (Dict[str, ndarray]): Dictionary of embeddings to be saved.
            last_save_id (int): The last save ID used for tracking saved embeddings.
            embeddings_file_path (Path): The path where embeddings are saved.
            use_reduced_embeddings (bool): Flag to determine if embeddings should be reduced.

        Returns:
            out (Tuple[int, Dict[str, ndarray]]): Updated last_save_id and an empty embeddings dictionary.
        """
        if use_reduced_embeddings:
            embeddings = self._reduce_embeddings(embeddings, self._embedder)
        last_save_id = self._save_embeddings(save_id=last_save_id, embeddings=embeddings,
                                             embeddings_file_path=embeddings_file_path)
        del embeddings
        return last_save_id, {}

    @staticmethod
    def embeddings_dimensionality_reduction(
            embeddings: Dict[str, Any],
            dimension_reduction_method: str,
            n_reduced_components: int) -> Dict[str, Any]:
        """Reduces the dimension of per-protein embeddings using one of the
        dimensionality reduction methods

        Args:
            embeddings (Dict[str, Any]): Dictionary of embeddings.
            dimension_reduction_method (str): The method used to reduce 
            the dimensionality of embeddings. Options are 'umap' or 'tsne'.
            n_reduced_components (int): The target number of dimensions for 
            the reduced embeddings.

        Returns:
            Dict[str, Any]: Dictionary of embeddings with reduced dimensions.
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
    def _reduce_embeddings(embeddings: Dict[str, ndarray], embedder) -> Dict[str, ndarray]:
        """
        Reduces the per-residue embeddings to per-protein embeddings.

        Parameters:
            embeddings (Dict[str, ndarray]): Dictionary of embeddings.
            embedder: The embedder used for reducing embeddings.

        Returns:
            out (Dict[str, ndarray]): Dictionary of reduced embeddings.
        """
        return {seq_id: embedder.reduce_per_protein(embedding) for seq_id, embedding in
                embeddings.items()}

    @staticmethod
    def _save_embeddings(save_id: int, embeddings: Dict[str, ndarray], embeddings_file_path: Path) -> int:
        """
        Saves the embeddings to a file.

        Args:
            save_id (int): The save ID used for tracking saved embeddings.
            embeddings (Dict[str, ndarray]): Dictionary of embeddings to be saved.
            embeddings_file_path (Path): The path where embeddings are saved.

        Returns:
            out (int): The updated save ID.
        """
        with h5py.File(embeddings_file_path, "a") as embeddings_file:
            idx = save_id
            for seq_id, embedding in embeddings.items():
                embeddings_file.create_dataset(str(idx), data=embedding, compression="gzip", chunks=True)
                embeddings_file[str(idx)].attrs["original_id"] = seq_id  # Follows biotrainer & bio_embeddings standard
                idx += 1
            return idx

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
    def load_embeddings(embeddings_file_path: str) -> Dict[str, Any]:
        """
        Loads precomputed embeddings from a file.

        Parameters:
            embeddings_file_path (str): Path to the embeddings file.

        Returns:
            out (Dict[str, Any]): Dictionary mapping sequence IDs to embeddings.
        """
        # Load computed embeddings in .h5 file format
        logger.info(f"Loading embeddings from: {embeddings_file_path}")
        start = time.perf_counter()

        # Old version see:
        # https://stackoverflow.com/questions/48385256/optimal-hdf5-dataset-chunk-shape-for-reading-rows/48405220#48405220
        embeddings_file = h5py.File(embeddings_file_path, 'r')

        # "original_id" from embeddings file -> Embedding
        id2emb = {embeddings_file[idx].attrs["original_id"]: torch.tensor(np.array(embedding)) for (idx, embedding) in
                  embeddings_file.items()}

        # Logging
        logger.info(f"Read {len(id2emb)} entries.")
        logger.info(f"Time elapsed for reading embeddings: {(time.perf_counter() - start):.1f}[s]")

        return id2emb
