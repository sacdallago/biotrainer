import os
import gc
import time
import psutil
import h5py
import torch
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path
from numpy import ndarray
from typing import Dict, Tuple, Any, List, Optional

from .embedder_interfaces import EmbedderInterface

from ..protocols import Protocol
from ..utilities import read_FASTA

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    A service class for computing embeddings using a provided embedder.
    """

    def __init__(self, embedder: EmbedderInterface = None, use_half_precision: bool = False):
        self._embedder = embedder
        self._use_half_precision = use_half_precision

    def compute_embeddings(self, sequence_file: str, output_dir: Path, protocol: Protocol,
                           force_output_dir: Optional[bool] = False,
                           force_recomputing: Optional[bool] = False) -> str:
        """
        Compute embeddings with the provided embedder from a sequence file.

        Parameters:
            sequence_file (str): Path to the sequence file.
            output_dir (Path): Output directory to store the computed embeddings.
            protocol (Protocol): Protocol for the embeddings. Determines if the embeddings should be reduced to per-protein.
            force_output_dir (bool, optional): If True, the given output directory is directly used to store the embeddings file,
                without any path enhancement. Defaults to False.
            force_recomputing (bool, optional): If True, the embedding file is re-computed, even if it already exists. Defaults to False.

        Returns:
            str: Path to the generated output embeddings file.
        """
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

        # Avoid re-computation if file already exists
        if not force_recomputing and embeddings_file_path.is_file():
            return str(embeddings_file_path)

        logger.info(f"Computing embeddings to: {str(embeddings_file_path)}")

        protein_sequences = {seq.id: str(seq.seq) for seq in sorted(read_FASTA(sequence_file),
                                                                    key=lambda seq: len(seq.seq),
                                                                    reverse=True)}

        embeddings_file_path = self.embedding_service(protein_sequences, embeddings_file_path, use_reduced_embeddings)
        
        del self._embedder
        gc.collect()

        return str(embeddings_file_path)
    
    def embedding_service(self, protein_sequences: Dict[str, str], embeddings_file_path: Path,
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
        embeddings_on_ram: int = 0
        start_time = time.time()
        
        embedding_iter = self._embedder.embed_many(protein_sequences.values())
        total_sequences = len(protein_sequences.values())
        
        # Get the first embedding
        embeddings[sequence_ids[idx]] = next(embedding_iter, None)
        
        if embeddings[sequence_ids[idx]] is None:
            logger.info(f"No embeddings found.")
            return str(embeddings_file_path)
        
        logger.info(f"Embedding dimension: {embeddings[sequence_ids[idx]].shape[-1]}")
        logger.info("Checking for ultra-long reads...")
        
        # For ultra-long reads
        with tqdm(total=total_sequences, initial=idx, desc="Processing Sequences") as pbar:
            while True:
                max_embedding_fit = self._max_embedding_fit(embeddings[sequence_ids[idx]])
                logger.info(f"New {max_embedding_fit=}")
                embeddings_on_ram += 1 
                
                # if ultra-long, save it without loading any more
                if max_embedding_fit <= 3:
                    logger.info("Ultra-long read found, saving it...")
                    last_save_id, embeddings = self._save_and_reset_embeddings(embeddings, idx, last_save_id,
                                                                embeddings_file_path, use_reduced_embeddings)
                    embeddings_on_ram = 0
                    idx += 1
                    pbar.update(1)
                    if idx == len(sequence_ids):
                        logger.info("Processing all the reads are done.")
                        break
                    embeddings[sequence_ids[idx]] = next(embedding_iter, None)
                else:
                    logger.info(f"{idx} ultra-long reads found and processed.")
                    break
                

        # If we have reads left process the normal ones
        if idx != len(sequence_ids):
            logger.info(f"Processing normal reads...")
            with tqdm(total=total_sequences, initial=idx, desc="Processing Sequences") as pbar:
                while True:
                    idx += 1
                    pbar.update(1)
                    if idx == len(sequence_ids):
                        logger.info("Processing normal reads done.")
                        break
                    embeddings[sequence_ids[idx]] = next(embedding_iter, None)
                    embeddings_on_ram += 1
                    
                    if embeddings_on_ram % max_embedding_fit == 0 or idx == total_sequences:
                        max_embedding_fit = self._max_embedding_fit(embeddings[sequence_ids[idx]])
                        last_save_id, embeddings = self._save_and_reset_embeddings(embeddings, idx, last_save_id,
                                                                    embeddings_file_path, use_reduced_embeddings)
                        logger.info(f"New {max_embedding_fit=}")
                        embeddings_on_ram = 0
                    
        # Save remaining embeddings
        if len(embeddings) > 0:
            last_save_id, embeddings = self._save_and_reset_embeddings(embeddings, idx, last_save_id,
                                                            embeddings_file_path, use_reduced_embeddings)

        end_time = time.time()
        logger.info(f"Time elapsed for saving embeddings: {end_time - start_time:.2f}[s]")
        
        del embeddings
        gc.collect()

        return str(embeddings_file_path)

    @staticmethod
    def _max_embedding_fit(embedding: ndarray) -> int:
        """
        Calculates the maximum number of embeddings that can fit in available memory.

        Parameters:
            embedding (ndarray): An embedding array.

        Returns:
            int: The maximum number of embeddings that can fit in memory.
        """
        max_embedding_fit = int(0.75 * (psutil.virtual_memory().available / (embedding.size*18)))
        max_embedding_fit = 1 if max_embedding_fit == 0 else max_embedding_fit
        return max_embedding_fit
    
    def _save_and_reset_embeddings(self, embeddings: Dict[str, ndarray], idx: int, last_save_id: int,
                                   embeddings_file_path: Path, use_reduced_embeddings: bool) -> Tuple[int, Dict[str, ndarray]]:
        """
        Save the embeddings and reset the dictionary.

        Parameters:
            embeddings (Dict[str, ndarray]): Dictionary of embeddings to be saved.
            last_save_id (int): The last save ID used for tracking saved embeddings.
            embeddings_file_path (Path): The path where embeddings are saved.
            use_reduced_embeddings (bool): Flag to determine if embeddings should be reduced.
            idx (int): Current index, used for logging purposes.

        Returns:
            out (Tuple[int, Dict[str, ndarray]]): Updated last_save_id and an empty embeddings dictionary.
        """
        if use_reduced_embeddings:
            embeddings = self._reduce_embeddings(embeddings, self._embedder)
        last_save_id = self._save_embeddings(save_id=last_save_id, embeddings=embeddings,
                                            embeddings_file_path=embeddings_file_path)
        logger.info(f"Saving until index {idx}")
        del embeddings
        return last_save_id, {}

    
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

        embeddings = list(
            tqdm(self._embedder.embed_many(protein_sequences), total=len(protein_sequences)))

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
