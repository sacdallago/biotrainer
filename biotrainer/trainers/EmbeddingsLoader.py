import h5py
import time
import torch
import logging

from pathlib import Path
from bio_embeddings.utilities.pipeline import execute_pipeline_from_config

# Defines if reduced embeddings from bio_embeddings should be used.
# Reduced means that the per-residue embeddings are reduced to a per-sequence embedding
_PROTOCOL_TO_REDUCED_EMBEDDINGS = {
    "residue_to_class": False,
    "residues_to_class": False,
    "sequence_to_class": True,
}

logger = logging.getLogger(__name__)


class EmbeddingsLoader:

    def __init__(self,
                 # Necessary
                 protocol: str, embedder_name: str, sequence_file: str, output_dir: Path,
                 # Optional (only for precomputed embeddings)
                 embeddings_file_path: str = None,
                 **kwargs):
        self._embedder_name = embedder_name
        self._sequence_file = sequence_file
        self._output_dir = output_dir
        self._embeddings_file_path = embeddings_file_path
        self._use_reduced_embeddings = _PROTOCOL_TO_REDUCED_EMBEDDINGS[protocol]

    def load_embeddings(self):
        # If embeddings don't exist, create them using the bio_embeddings pipeline
        if not self._embeddings_file_path or not Path(self._embeddings_file_path).is_file():
            embeddings_config = {
                "global": {
                    "sequences_file": self._sequence_file,
                    "prefix": str(self._output_dir / self._embedder_name),
                    "simple_remapping": True
                },
                "embeddings": {
                    "type": "embed",
                    "protocol": self._embedder_name,
                    "reduce": self._use_reduced_embeddings,
                    "discard_per_amino_acid_embeddings": self._use_reduced_embeddings
                }
            }
            embeddings_file_name = "reduced_embeddings_file.h5" \
                if self._use_reduced_embeddings else "embeddings_file.h5"
            # Check if bio-embeddings has already been run
            self._embeddings_file_path = str(
                Path(embeddings_config['global']['prefix']) / "embeddings" / embeddings_file_name)

            if not Path(self._embeddings_file_path).is_file():
                _ = execute_pipeline_from_config(embeddings_config, overwrite=False)

        # load pre-computed embeddings in .h5 file format computed via bio_embeddings
        logger.info(f"Loading embeddings from: {self._embeddings_file_path}")
        start = time.time()

        # https://stackoverflow.com/questions/48385256/optimal-hdf5-dataset-chunk-shape-for-reading-rows/48405220#48405220
        embeddings_file = h5py.File(self._embeddings_file_path, 'r', rdcc_nbytes=1024 ** 2 * 4000, rdcc_nslots=1e7)
        id2emb = {embeddings_file[idx].attrs["original_id"]: embedding for (idx, embedding) in
                  embeddings_file.items()}
        embeddings_length = list(id2emb.values())[0].shape[-1]  # Last position in shape is always embedding length
        # Logging
        logger.info(f"Read {len(id2emb)} entries.")
        logger.info(f"Time elapsed for reading embeddings: {(time.time() - start):.1f}[s]")
        return id2emb, embeddings_length
