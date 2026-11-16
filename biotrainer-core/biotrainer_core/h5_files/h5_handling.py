import io
import h5py
import base64

from pathlib import Path
from typing import Optional, Set, Dict, Any, Union, List

from ..input_files import read_FASTA
from ..data_classes import SequenceData


def read_id2emb(h5_file_path: Union[str, Path], ids_to_load: Optional[Set[str]] = None) -> Dict[str, Any]:
    """ Read raw content from h5_file_path and return as a dictionary """
    # Old version see:
    # https://stackoverflow.com/questions/48385256/optimal-hdf5-dataset-chunk-shape-for-reading-rows/48405220#48405220
    # Sequence hash from embeddings file -> Embedding
    id2emb = {}
    with h5py.File(h5_file_path, 'r') as embeddings_file:
        if ids_to_load is None:
            # Load all sequences
            id2emb = {idx: embedding[:] for (idx, embedding) in embeddings_file.items()}
        else:
            for idx in ids_to_load:
                if idx in embeddings_file:
                    id2emb[idx] = embeddings_file[idx][:]
                else:
                    raise ValueError(f"Sequence hash {idx} not found in embeddings file!")
    return id2emb


def store_embedding_to_path(h5_file_path: Union[str, Path], seq_hash: str, seq_id: str, embedding_data) -> bool:
    """ Store embedding in h5_file (can be Path or str) """
    with h5py.File(h5_file_path, "a") as h5_file_handle:
        store_embedding_to_handle(h5_file_handle, seq_hash, seq_id, embedding_data)
    return True


def store_embedding_to_handle(h5_file_handle: h5py.File, seq_hash: str, seq_id: str, embedding_data):
    """ Store embedding in h5_file via h5py.File handle """
    h5_file_handle.create_dataset(seq_hash, data=embedding_data)
    h5_file_handle[seq_hash].attrs["original_id"] = seq_id
    return True


def export_sequence_data_with_embeddings(embd_records: List[SequenceData]) -> str:
    """ Export sequence data with embeddings to a base64-encoded h5 file string """
    h5_io = io.BytesIO()
    with h5py.File(h5_io, "w") as embeddings_file:
        for embd_record in embd_records:
            seq_hash = embd_record.get_hash()
            store_embedding_to_handle(embeddings_file, seq_hash, embd_record.seq_id, embd_record.embedding)

    h5_io.seek(0)
    h5_base64 = base64.b64encode(h5_io.getvalue()).decode("utf-8")
    h5_io.close()
    return h5_base64


def filter_h5_with_fasta(h5_file_path: Union[str, Path], fasta_file_path: Union[str, Path],
                         h5_output_path: Union[str, Path]) -> int:
    """
    Filter h5_file_path to only contain sequences/seq_ids present in FASTA file

    :returns number of entries kept
    """
    seq_data = read_FASTA(fasta_file_path)
    if len(seq_data) == 0:
        raise ValueError(f"No sequences found in FASTA file {fasta_file_path}!")
    seq_ids = {seq_record.seq_id for seq_record in seq_data
               if seq_record.seq_id is not None and len(seq_record.seq_id) > 0}
    seq_hashes = {seq_record.get_hash() for seq_record in seq_data}
    combined = seq_ids | seq_hashes
    n_kept = 0
    with h5py.File(h5_file_path, "r+") as h5_input_file:
        with h5py.File(h5_output_path, "a") as h5_output_file:
            for key in h5_input_file.keys():
                if key in combined:
                    store_embedding_to_handle(h5_output_file, key,
                                              h5_input_file[key].attrs["original_id"],
                                              h5_input_file[key][:])
                    n_kept += 1
    return n_kept
