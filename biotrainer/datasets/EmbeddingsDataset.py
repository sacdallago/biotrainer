import torch

from typing import Tuple, List

from ..utilities import DatasetSample


class __EmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[DatasetSample]):
        self.ids, self.inputs, self.targets = zip(
            *[(sample.seq_id, sample.embedding, sample.target) for sample in samples]
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.LongTensor]:
        seq_id = self.ids[index]
        x = self.inputs[index].float()
        y = self.targets[index].long()
        return seq_id, x, y


class ResidueEmbeddingsClassificationDataset(__EmbeddingsDataset):
    pass


class SequenceEmbeddingsClassificationDataset(__EmbeddingsDataset):
    pass

""" TODO
class SequenceEmbeddingsInteractionDataset(__EmbeddingsDataset):
    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.LongTensor]:
        seq_id = self.ids[index]
        x = self.inputs[index].float()
        y = self.targets[index].float()
        return seq_id, x, y
"""

class SequenceEmbeddingsRegressionDataset(__EmbeddingsDataset):
    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.LongTensor]:
        seq_id = self.ids[index]
        x = self.inputs[index].float()
        y = self.targets[index].float()
        return seq_id, x, y
