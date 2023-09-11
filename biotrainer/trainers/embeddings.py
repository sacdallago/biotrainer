import os
import time
import h5py
import torch
import logging
import numpy as np

from pathlib import Path
from typing import Dict, Any
from importlib.util import spec_from_file_location, module_from_spec

from .custom_embedder import CustomEmbedder

from ..protocols import Protocol
from ..config import ConfigurationException

# Defines if reduced embeddings from bio_embeddings should be used.
# Reduced means that the per-residue embeddings are reduced to a per-sequence embedding
_REQUIRES_REDUCED_EMBEDDINGS = {
    Protocol.residue_to_class: False,
    Protocol.residues_to_class: False,
    Protocol.sequence_to_class: True,
    Protocol.sequence_to_value: True
}

logger = logging.getLogger(__name__)


def compute_embeddings(embedder_name: str, sequence_file: str, output_dir: Path, protocol: Protocol) -> str:
    # Create protocol path to embeddings, because bio-embeddings can't handle recursive dir creation
    output_dir /= protocol.name
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if embedder_name[-3:] == ".py":  # Check that file ends on .py => Python script
        if Path(embedder_name).exists():
            if Path(embedder_name).is_file():
                return _compute_via_custom_embedder(embedder_name=embedder_name, sequence_file=sequence_file,
                                                    output_dir=output_dir, protocol=protocol)
            else:
                raise Exception(f"Custom embedder should be used, but path to script is not a file!\n"
                                f"embedder_name: {embedder_name}")
        else:
            raise Exception(f"Custom embedder should be used, but path to script does not exist!\n"
                            f"embedder_name: {embedder_name}")
    else:
        return _compute_via_bio_embeddings(embedder_name=embedder_name, sequence_file=sequence_file,
                                           output_dir=output_dir, protocol=protocol)


def _compute_via_custom_embedder(embedder_name: str, sequence_file: str, output_dir: Path, protocol: Protocol) -> str:
    # Load the module from the file path
    spec = spec_from_file_location("module_name", embedder_name)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check for some problematic modules from external script
    disallow_modules = ["requests", "urllib", "os"]
    for name in dir(module):
        if name in disallow_modules:
            raise Exception(f"Module {name} not allowed for custom embedder script!")

    # Find custom embedder in script
    custom_embedder = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, CustomEmbedder) and not obj.__name__ == "CustomEmbedder":
            logger.info(f"Using custom embedder: {obj.__name__}")
            custom_embedder = obj
            break

    if custom_embedder is None:
        raise Exception(f"Did not find custom embedder {embedder_name} in the provided script!")

    custom_embedder_instance = custom_embedder()
    use_reduced_embeddings = _REQUIRES_REDUCED_EMBEDDINGS[protocol]

    output_dir /= custom_embedder.name
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir /= ("reduced_" if use_reduced_embeddings else "") + f"embeddings_file_{custom_embedder.name}.h5"

    return custom_embedder_instance.embed_many(sequence_file=sequence_file, output_path=str(output_dir),
                                               reduce_per_protein=use_reduced_embeddings)


def _compute_via_bio_embeddings(embedder_name: str, sequence_file: str, output_dir: Path, protocol: Protocol) -> str:
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
            'prefix': str(output_dir / embedder_name),
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
    id2emb = {embeddings_file[idx].attrs["original_id"]: torch.tensor(np.array(embedding)) for (idx, embedding) in
              embeddings_file.items()}

    # Logging
    logger.info(f"Read {len(id2emb)} entries.")
    logger.info(f"Time elapsed for reading embeddings: {(time.perf_counter() - start):.1f}[s]")

    return id2emb
