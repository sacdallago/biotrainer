# H5 File Standardization

This guide explains how biotrainer writes and reads H5 embedding files, how dataset keys are chosen, and why sequence
hashes are used by default.

## What goes into the H5 file

- One dataset (per-residue or per-sequence embedding) per input sequence stored under 
a dataset key (also called index) inside the H5 file.
- Each dataset stores the original FASTA header ID as an attribute named `original_id`.
    - Code reference: `biotrainer/embedders/services/embedding_service.py::EmbeddingService.store_embedding`
    - Implementation:
        - Dataset creation: `create_dataset(<key>, data=<embedding>, compression="gzip", chunks=True)`
        - Attribute: `embeddings_file[<key>].attrs["original_id"] = <seq_id>`

### Embedding shapes

- Per-sequence embeddings: `num_sequences x embedding_dim`
- Per-residue embeddings: `num_sequences x seq_len x embedding_dim`
    - `embedding_dim` is fixed across sequences; `seq_len` can vary per sequence.

## How dataset keys are chosen

- By default, biotrainer uses a sequence hash as the H5 dataset key. This behavior is controlled by `store_by_hash=True`
  in `EmbeddingService.compute_embeddings(...)`.
- The original FASTA ID is always preserved as the `original_id` attribute for traceability.

Code references:

- Key selection and writing: `biotrainer/embedders/services/embedding_service.py`
    - `_compute_embeddings_parallel` and `store_embedding` use `seq_record.get_hash()` when `store_by_hash=True`.
- Hash computation: `biotrainer/utilities/hashing.py::calculate_sequence_hash`
- Sequence record wrapper: `biotrainer/input_files/biotrainer_seq_record.py::BiotrainerSequenceRecord.get_hash`

## Sequence hash definition (formula)

The sequence hash is computed as SHA-256 over the raw sequence concatenated with an underscore and its length.

Python reference (exact implementation):

```python
import hashlib


def calculate_sequence_hash(sequence: str) -> str:
    suffix = len(sequence)
    sequence = f"{sequence}_{suffix}"
    return hashlib.sha256(sequence.encode()).hexdigest()
```

## Why hashes (instead of FASTA IDs)

- Unambiguous and deterministic
    - FASTA IDs are often reused, truncated, or modified across datasets, leading to collisions.
    - The hash depends on the actual sequence content (+ length), uniquely identifying that exact sequence across files
      and runs.
- H5 key safety
    - HDF5 dataset names must not contain certain characters (e.g., `/`). FASTA IDs sometimes do. Hex hashes are always
      safe.
- Efficient caching and avoiding recomputation
    - Content hashes enable quick existence checks across runs/files. If a sequence was embedded before, the same hash
      key can be reused, avoiding expensive recomputation.
- Easier merging/alignment
    - Multiple sources containing the same sequence resolve to the same key, simplifying joins and integrity checks.



    

