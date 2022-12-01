from .collate_functions import pad_sequence_embeddings, pad_residue_embeddings, pad_residues_embeddings
from .EmbeddingsDataset import ResidueEmbeddingsClassificationDataset, SequenceEmbeddingsClassificationDataset, \
    SequenceEmbeddingsInteractionDataset, SequenceEmbeddingsRegressionDataset

__DATASETS = {
    'residue_to_class': ResidueEmbeddingsClassificationDataset,
    'residues_to_class': ResidueEmbeddingsClassificationDataset,
    'sequence_to_class': SequenceEmbeddingsClassificationDataset,
    'sequence_to_value': SequenceEmbeddingsRegressionDataset,
    'protein_protein_interaction': SequenceEmbeddingsInteractionDataset
}

__COLLATE_FUNCTIONS = {
    'residue_to_class': pad_residue_embeddings,
    'residues_to_class': pad_residues_embeddings,
    'sequence_to_class': pad_sequence_embeddings,
    'sequence_to_value': pad_sequence_embeddings,
    'protein_protein_interaction': pad_sequence_embeddings,
}


def get_dataset(protocol: str, samples: dict):
    dataset = __DATASETS.get(protocol)

    if not dataset:
        raise NotImplementedError
    else:
        return dataset(samples=samples)


def get_collate_function(protocol: str):
    collate_function = __COLLATE_FUNCTIONS.get(protocol)

    return collate_function


__all__ = [
    'get_collate_function',
    'get_dataset',
]
