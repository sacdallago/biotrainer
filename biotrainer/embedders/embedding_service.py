import os
import time
import h5py
import torch
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

from .embedder_interfaces import EmbedderInterface

from ..protocols import Protocol
from ..utilities import read_FASTA

# Defines if reduced embeddings should be used.
# Reduced means that the per-residue embeddings are reduced to a per-sequence embedding
_REQUIRES_REDUCED_EMBEDDINGS = {
    Protocol.residue_to_class: False,
    Protocol.residues_to_class: False,
    Protocol.sequence_to_class: True,
    Protocol.sequence_to_value: True
}

logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(self, embedder: EmbedderInterface = None, use_half_precision: bool = False):
        self._embedder = embedder
        self._use_half_precision = use_half_precision

    def compute_embeddings(self, sequence_file: str, output_dir: Path, protocol: Protocol) -> str:
        # Create protocol path to embeddings
        embeddings_file_path = output_dir / protocol.name
        if not os.path.isdir(embeddings_file_path):
            os.mkdir(embeddings_file_path)

        use_reduced_embeddings = _REQUIRES_REDUCED_EMBEDDINGS[protocol]

        embedder_name = self._embedder.name.split("/")[-1]
        embeddings_file_path /= embedder_name
        if not os.path.isdir(embeddings_file_path):
            os.mkdir(embeddings_file_path)
        embeddings_file_path /= (("reduced_" if use_reduced_embeddings else "")
                                 + f"embeddings_file_{embedder_name}{'_half' if self._use_half_precision else ''}.h5")

        # Avoid re-computation if file already exists
        if embeddings_file_path.is_file():
            return str(embeddings_file_path)

        logger.info(f"Computing embeddings to: {str(embeddings_file_path)}")

        protein_sequences = {seq.id: str(seq.seq) for seq in sorted(read_FASTA(sequence_file),
                                                                    key=lambda seq: len(seq.seq),
                                                                    reverse=True)}

        embeddings = list(
            tqdm(self._embedder.embed_many(protein_sequences.values()), total=len(protein_sequences.values())))

        if use_reduced_embeddings:
            embeddings = [self._embedder.reduce_per_protein(embedding) for embedding in embeddings]

        with h5py.File(embeddings_file_path, "w") as embeddings_file:
            idx = 0
            for seq_id, embedding in zip(protein_sequences.keys(), embeddings):
                embeddings_file.create_dataset(str(idx), data=embedding, compression="gzip", chunks=True)
                embeddings_file[str(idx)].attrs["original_id"] = seq_id  # Follows biotrainer & bio_embeddings standard
                idx += 1

        return str(embeddings_file_path)

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
