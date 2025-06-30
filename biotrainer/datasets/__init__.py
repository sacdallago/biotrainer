from typing import List

from .collate_functions import pad_sequence_embeddings, pad_residue_embeddings, pad_residues_embeddings
from .embeddings_dataset import SequenceDataset, EmbeddingsDataset, BiotrainerDataset

from ..protocols import Protocol


__COLLATE_FUNCTIONS = {
    Protocol.residue_to_class: pad_residue_embeddings,
    Protocol.residues_to_class: pad_residues_embeddings,
    Protocol.residues_to_value: pad_residues_embeddings,
    Protocol.sequence_to_class: pad_sequence_embeddings,
    Protocol.sequence_to_value: pad_sequence_embeddings,
}


def get_dataset(samples: List, finetuning: bool):
    if finetuning:
        return SequenceDataset(samples=samples)

    return EmbeddingsDataset(samples=samples)


def get_embeddings_collate_function(protocol: Protocol):
    collate_function = __COLLATE_FUNCTIONS.get(protocol)

    return collate_function


__all__ = [
    'get_embeddings_collate_function',
    'get_dataset',
    'BiotrainerDataset'
]
