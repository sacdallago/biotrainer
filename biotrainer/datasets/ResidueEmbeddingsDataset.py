import torch

from typing import Tuple


class ResidueEmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
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
