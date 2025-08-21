from typing import List, Dict, Optional

from .collate_functions import pad_sequence_embeddings, pad_residue_embeddings, pad_residues_embeddings
from .embeddings_dataset import SequenceDataset, EmbeddingsDataset, BiotrainerDataset, SequenceDatasetWithRandomMasking

from ..protocols import Protocol

__COLLATE_FUNCTIONS = {
    Protocol.residue_to_class: pad_residue_embeddings,
    Protocol.residue_to_value: pad_residue_embeddings,
    Protocol.residues_to_class: pad_residues_embeddings,
    Protocol.residues_to_value: pad_residues_embeddings,
    Protocol.sequence_to_class: pad_sequence_embeddings,
    Protocol.sequence_to_value: pad_sequence_embeddings,
}


def get_dataset(samples: List, finetuning: bool, random_masking: bool, mask_token: Optional[str],
                class_str2int: Optional[Dict[str, int]], ):
    if finetuning:
        if random_masking:
            return SequenceDatasetWithRandomMasking(samples=samples, mask_token=mask_token, class_str2int=class_str2int)

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
