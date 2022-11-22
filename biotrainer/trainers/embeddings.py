import os
import time
import h5py
import torch
import shutil
import logging

from pathlib import Path
from urllib import request
from typing import Dict, Any

from ..utilities.config import ConfigurationException

# Defines if reduced embeddings from bio_embeddings should be used.
# Reduced means that the per-residue embeddings are reduced to a per-sequence embedding
_REQUIRES_REDUCED_EMBEDDINGS = {
    'residue_to_class': False,
    'residues_to_class': False,
    'sequence_to_class': True,
    'sequence_to_value': True,
    'protein_protein_interaction': True
}

logger = logging.getLogger(__name__)


def download_embeddings(url: str, script_path: str) -> str:
    try:
        logger.info(f"Trying to download embeddings from {url}")
        req = request.Request(url, headers={
            'User-Agent': "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"
        })

        file_name = url.split("/")[-1]
        save_path = script_path + "/downloaded_" + file_name
        with request.urlopen(req) as response, open(save_path, 'wb') as outfile:
            shutil.copyfileobj(response, outfile)
        logger.info(f"Embeddings file successfully downloaded and stored at {save_path}.")
        return save_path
    except Exception as e:
        raise Exception(f"Could not download embeddings from url {url}") from e


def compute_embeddings(embedder_name: str, sequence_file: str, output_dir: Path, protocol: str) -> str:
    use_reduced_embeddings = _REQUIRES_REDUCED_EMBEDDINGS[protocol]

    try:
        from bio_embeddings.utilities.pipeline import execute_pipeline_from_config
    except ImportError:
        raise ConfigurationException(
            f"Trying to compute non-existing embeddings without bio-embeddings installed. "
            "Install via `poetry install --extras \"bio-embeddings\"`"
        )
    embeddings_config = {
        'global': {
            'sequences_file': sequence_file,
            'prefix': str(output_dir / protocol / embedder_name),
            'simple_remapping': True
        },
        'embeddings': {
            'type': 'embed',
            'protocol': embedder_name,
            'reduce': use_reduced_embeddings,
            'discard_per_amino_acid_embeddings': use_reduced_embeddings
        }
    }
    embeddings_file_name = "reduced_embeddings_file.h5" \
        if use_reduced_embeddings else "embeddings_file.h5"

    # Create protocol path to embeddings, because bio-embeddings can't handle recursive dir creation
    if not os.path.isdir(output_dir / protocol):
        os.mkdir(output_dir / protocol)

    # Check if bio-embeddings has already been run
    embeddings_file_path = str(Path(embeddings_config['global']['prefix']) / "embeddings" / embeddings_file_name)

    if not Path(embeddings_file_path).is_file():
        try:
            _ = execute_pipeline_from_config(embeddings_config, overwrite=False)
        except FileExistsError as e:
            raise FileExistsError(f"The directory for {embedder_name} does already exist, "
                                  f"but no existing embeddings have been found. Something must have gone wrong "
                                  f"before, try to delete the output directory and restart.") from e
    return embeddings_file_path


def load_embeddings(embeddings_file_path: str) -> Dict[str, Any]:
    # load pre-computed embeddings in .h5 file format computed via bio_embeddings
    logger.info(f"Loading embeddings from: {embeddings_file_path}")
    start = time.perf_counter()

    # Old version see:
    # https://stackoverflow.com/questions/48385256/optimal-hdf5-dataset-chunk-shape-for-reading-rows/48405220#48405220
    embeddings_file = h5py.File(embeddings_file_path, 'r')

    # "original_id" from embeddings file -> Embedding
    id2emb = {embeddings_file[idx].attrs["original_id"]: torch.tensor(embedding) for (idx, embedding) in
              embeddings_file.items()}

    # Logging
    logger.info(f"Read {len(id2emb)} entries.")
    logger.info(f"Time elapsed for reading embeddings: {(time.perf_counter() - start):.1f}[s]")

    return id2emb
