import torch

from typing import Tuple


class __EmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, samples: dict):
        self.ids, self.inputs, self.targets = zip(
            *[(seq_id, inputs, targets) for seq_id, (inputs, targets) in samples.items()]
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


class SequenceEmbeddingsInteractionDataset(__EmbeddingsDataset):
    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.LongTensor]:
        seq_id = self.ids[index]
        x = self.inputs[index].float()
        y = self.targets[index].float()
        return seq_id, x, y


class SequenceEmbeddingsRegressionDataset(__EmbeddingsDataset):
    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.LongTensor]:
        seq_id = self.ids[index]
        x = self.inputs[index].float()
        y = self.targets[index].float()
        return seq_id, x, y
