import torch

from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence

from ..utilities import SEQUENCE_PAD_VALUE, MASK_AND_LABELS_PAD_VALUE


def pad_sequence_embeddings(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                            ) -> Tuple[List, torch.Tensor, torch.Tensor, None]:
    ids = [x[0] for x in batch]
    x = [x[1] for x in batch]
    y = [x[2] for x in batch]

    return ids, torch.stack(x), torch.stack(y), None


def pad_residue_embeddings(
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], batch_first=True
) -> Tuple[List, torch.Tensor, torch.Tensor, torch.LongTensor]:
    """

    :param batch_first: first dimension of return value will be batch
    :param batch: An iteration batch from the ResidueEmbeddingsDataset class [(ID, embedding, label)]
    :return: Returns padded embeddings, labels, lengths and True/False mask for padding of input embeddings.
    """

    # Each element in "batch" is a tuple [(ID, embedding, target)]

    # Handle empty batch
    if not batch:
        return [], torch.tensor([]), torch.tensor([]), torch.LongTensor([])

    # Sort the batch in descending order of sequence length
    # Handle potential dimension issues
    try:
        sorted_batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)
    except IndexError:
        # If x[1] is 1D, or if there's only one item, don't sort
        sorted_batch = batch

    # Sequence ids:
    seq_ids = [x[0] for x in sorted_batch]

    # Get each residue-embedding and pad it
    sequences = [x[1] for x in sorted_batch]

    # Ensure all sequences are at least 2D
    sequences = [s if s.dim() > 1 else s.unsqueeze(0) for s in sequences]

    padded_sequences = pad_sequence(sequences, batch_first=batch_first, padding_value=SEQUENCE_PAD_VALUE)

    # Also need to store the original length of each sequence
    # This is later needed in order to unpad the sequences
    lengths = torch.LongTensor([len(x) for x in sequences])

    # Don't forget to grab the labels of the *sorted* batch
    padded_labels = pad_sequence([x[2] for x in sorted_batch],
                                 batch_first=batch_first, padding_value=MASK_AND_LABELS_PAD_VALUE)

    return seq_ids, padded_sequences, padded_labels, lengths


def pad_residues_embeddings(
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], batch_first=True
) -> Tuple[List, torch.Tensor, torch.Tensor, torch.LongTensor]:
    """

    :param batch_first: first dimension of return value will be batch
    :param batch: An iteration batch from the ResidueEmbeddingsDataset class [(ID, embedding, label)]
    :return: Returns padded embeddings, labels, lengths and True/False mask for padding of input embeddings.
    """

    # Each element in "batch" is a tuple [(ID, embedding, target)]

    # Handle empty batch
    if not batch:
        return [], torch.tensor([]), torch.tensor([]), torch.LongTensor([])

    # Sort the batch in descending order of sequence length
    # Handle potential dimension issues
    try:
        sorted_batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)
    except IndexError:
        # If x[1] is 1D, or if there's only one item, don't sort
        sorted_batch = batch

    # Sequence ids:
    seq_ids = [x[0] for x in sorted_batch]

    # Get each residue-embedding and pad it
    sequences = [x[1] for x in sorted_batch]

    # Ensure all sequences are at least 2D
    sequences = [s if s.dim() > 1 else s.unsqueeze(0) for s in sequences]

    padded_sequences = pad_sequence(sequences, batch_first=batch_first, padding_value=SEQUENCE_PAD_VALUE)

    # Also need to store the length of each sequence
    # This is later needed in order to unpad the sequences
    lengths = torch.LongTensor([len(x) for x in sequences])

    # Don't forget to grab the labels of the *sorted* batch
    labels = torch.stack([x[2] for x in sorted_batch])

    # Permute the embeddings to fit the LA architecture
    return seq_ids, padded_sequences, labels, lengths
