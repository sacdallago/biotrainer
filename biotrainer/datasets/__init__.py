from .collate_functions import pad_sequences
from .EmbeddingsDataset import ResidueEmbeddingsDataset, SequenceEmbeddingsDataset

__DATASETS = {
    'residue_to_class': ResidueEmbeddingsDataset,
    'sequence_to_class': SequenceEmbeddingsDataset
}


def get_dataset(protocol: str, samples: dict):
    dataset = __DATASETS.get(protocol)

    if not dataset:
        raise NotImplementedError
    else:
        return dataset(samples=samples)


__all__ = [
    'pad_sequences',
    'get_dataset',
]
