import torch.utils.data import DataLoader

from .TrainingDataLoader import TrainingDatasetLoader, ResidueEmbeddingsDataset
from .collate_functions import pad_sequences


def get_dataloader(customdata, batch_size: int, shuffle: bool = True, collate_fn: callable = None):
    """
    Returns a DataLoader for the given customdata.
    """
    if collate_fn is None:
        collate_fn = pad_sequences

    # Create dataloaders with collate function
    dataset = CustomDataset(customdata)

    return DataLoader(dataset=customdata, batch_size=batch_size, shuffle=shuffle, drop_last=False, collate_fn=collate_fn)



__all__ = [
    'TrainingDatasetLoader',
    'ResidueEmbeddingsDataset',
]