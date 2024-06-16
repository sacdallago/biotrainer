from typing import List

from .collate_functions import pad_sequence_embeddings, pad_residue_embeddings, pad_residues_embeddings
from .embeddings_dataset import ResidueEmbeddingsClassificationDataset, ResidueEmbeddingsRegressionDataset, \
    SequenceEmbeddingsClassificationDataset, SequenceEmbeddingsRegressionDataset

from ..protocols import Protocol

__DATASETS = {
    Protocol.residue_to_class: ResidueEmbeddingsClassificationDataset,
    Protocol.residues_to_class: ResidueEmbeddingsClassificationDataset,
    Protocol.residues_to_value: ResidueEmbeddingsRegressionDataset,
    Protocol.sequence_to_class: SequenceEmbeddingsClassificationDataset,
    Protocol.sequence_to_value: SequenceEmbeddingsRegressionDataset,
}

__COLLATE_FUNCTIONS = {
    Protocol.residue_to_class: pad_residue_embeddings,
    Protocol.residues_to_class: pad_residues_embeddings,
    Protocol.residues_to_value: pad_residues_embeddings,
    Protocol.sequence_to_class: pad_sequence_embeddings,
    Protocol.sequence_to_value: pad_sequence_embeddings,
}


def get_dataset(protocol: Protocol, samples: List):
    dataset = __DATASETS.get(protocol)

    if not dataset:
        raise NotImplementedError
    else:
        return dataset(samples=samples)


def get_collate_function(protocol: Protocol):
    collate_function = __COLLATE_FUNCTIONS.get(protocol)

    return collate_function


__all__ = [
    'get_collate_function',
    'get_dataset',
]
