import torch

from typing import Tuple


class SequenceEmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, samples:dict):
        self.ids = list()
        self.inputs = list()
        self.targets = list()
        for seq_id, (inputs, targets) in samples.items():
            self.ids.append(seq_id)
            self.inputs.append(torch.mean(inputs, dim=0))
            self.targets.append(targets)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor, torch.LongTensor]:
        seq_id = self.ids[index]
        x = self.inputs[index].float()
        y = self.targets[index].long()
        return seq_id, x, y
