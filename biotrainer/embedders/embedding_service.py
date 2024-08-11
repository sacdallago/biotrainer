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
from typing import Dict, Any, List, Optional

from .embedder_interfaces import EmbedderInterface

from ..protocols import Protocol
from ..utilities import read_FASTA, SAVE_AFTER_N_EMBEDDINGS

logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(self, embedder: EmbedderInterface = None, use_half_precision: bool = False):
        self._embedder = embedder
        self._use_half_precision = use_half_precision

    def compute_embeddings(self, sequence_file: str, output_dir: Path, protocol: Protocol,
                           force_output_dir: Optional[bool] = False,
                           force_recomputing: Optional[bool] = False) -> str:
        """
        Compute embeddings with the provided embedder from file.

        :param sequence_file: Path to the sequence file
        :param output_dir: Output directory to store the computed embeddings
        :param protocol: Protocol for the embeddings. Determines if the embeddings should be reduced to per-protein
        :param force_output_dir: If True, the given output directory is directly used to store the embeddings file,
            without any path enhancing
        :param force_recomputing: If True, the embedding file is re-recomputed, even if it already exists
        :return: Path to the generated output embeddings file
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
        sequence_ids = list(protein_sequences.keys())
        embeddings = {}
        idx = 0
        last_save_id = 0
        embeddings_on_ram = 0
        
        start_time = time.time()
        
        embedding_iter = self._embedder.embed_many(protein_sequences.values())
        total_sequences = len(protein_sequences.values())
        first_embedding = next(embedding_iter, None)
        
        if first_embedding is None:
            return str(embeddings_file_path)  # No sequences to process.

        embeddings[sequence_ids[idx]] = first_embedding
        idx += 1
        
        max_embedding_fit = int(0.8 * (psutil.virtual_memory().available / (len(first_embedding)*4)))
        logger.info(f"First {max_embedding_fit=}")
        
        for embedding in tqdm(embedding_iter, initial=1, total=total_sequences):
            embeddings[sequence_ids[idx]] = embedding
            idx += 1
            embeddings_on_ram += 1
            
            if embeddings_on_ram % max_embedding_fit == 0 or idx == total_sequences:
                if use_reduced_embeddings:
                    embeddings = self._reduce_embeddings(embeddings, self._embedder)
                last_save_id = self._save_embeddings(save_id=last_save_id, embeddings=embeddings,
                                                    embeddings_file_path=embeddings_file_path)
                logger.info(f"Saving until index {idx}")
                del embeddings
                embeddings = {}
                
                if idx < total_sequences:
                    next_embedding = next(embedding_iter, None)
                    if next_embedding is None:
                        break  # No more embeddings to process.
                    embeddings[sequence_ids[idx]] = next_embedding
                    max_embedding_fit = int(0.8 * (psutil.virtual_memory().available / (len(next_embedding)*4)))
                    embeddings_on_ram = 0
                    idx += 1
                    logger.info(f"New {max_embedding_fit=}")

        end_time = time.time()
        logger.info(f"Total time to load and embeddings: {end_time - start_time:.2f} seconds")
        
        # Save remaining embeddings
        if len(embeddings) > 0:
            if use_reduced_embeddings:
                embeddings = self._reduce_embeddings(embeddings, self._embedder)
            _ = self._save_embeddings(save_id=last_save_id, embeddings=embeddings,
                                      embeddings_file_path=embeddings_file_path)

        # Delete embeddings and embedding model from memory now, because they will no longer be needed
        del embeddings
        del self._embedder
        gc.collect()

        return str(embeddings_file_path)

    
    @staticmethod
    def _reduce_embeddings(embeddings: Dict[str, ndarray], embedder) -> Dict[str, ndarray]:
        return {seq_id: embedder.reduce_per_protein(embedding) for seq_id, embedding in
                embeddings.items()}

    @staticmethod
    def _save_embeddings(save_id: int, embeddings: Dict[str, ndarray], embeddings_file_path: Path) -> int:
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

        :param protein_sequences: List of protein sequences as string
        :param protocol: Protocol for the embeddings. Determines if the embeddings should be reduced to per-protein
        :return: List of computed embeddings
        """
        use_reduced_embeddings = protocol in Protocol.using_per_sequence_embeddings()

        embeddings = list(
            tqdm(self._embedder.embed_many(protein_sequences), total=len(protein_sequences)))

        if use_reduced_embeddings:
            embeddings = [self._embedder.reduce_per_protein(embedding) for embedding in embeddings]
        return embeddings

    @staticmethod
    def load_embeddings(embeddings_file_path: str) -> Dict[str, Any]:
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
