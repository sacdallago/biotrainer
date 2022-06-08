from .collate_functions import pad_sequences
from .EmbeddingsDataset import ResidueEmbeddingsDataset, SequenceEmbeddingsDataset, SequenceEmbeddingsRegressionDataset

__DATASETS = {
    'residue_to_class': ResidueEmbeddingsDataset,
    'sequence_to_class': SequenceEmbeddingsDataset,
    'sequence_to_value': SequenceEmbeddingsRegressionDataset
}

__COLLATE_FUNCTIONS = {
    'residue_to_class': pad_sequences,
    'sequence_to_class': None,
    'sequence_to_value': None
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
