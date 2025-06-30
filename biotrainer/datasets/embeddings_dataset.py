import torch

from typing import List, Union

from ..utilities import EmbeddingDatasetSample, SequenceDatasetSample


class BiotrainerDataset(torch.utils.data.Dataset):
    ids: List[str]
    inputs: Union[List[torch.tensor], List[str]]  # Embeddings / Sequences
    targets: List[torch.tensor]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        seq_id = self.ids[index]
        x = self.inputs[index]
        y = self.targets[index]
        return seq_id, x, y


class EmbeddingsDataset(BiotrainerDataset):
    """Embeddings dataset for training on pre-calculated embeddings."""
    def __init__(self, samples: List[EmbeddingDatasetSample]):
        self.ids, self.inputs, self.targets = zip(
            *[(sample.seq_id, sample.embedding, sample.target) for sample in samples]
        )


class SequenceDataset(BiotrainerDataset):
    """Sequence dataset for fine-tuning"""
    def __init__(self, samples: List[SequenceDatasetSample],):
        self.ids, self.inputs, self.targets = zip(
            *[(sample.seq_id, sample.seq_record.seq, sample.target) for sample in samples]
        )
