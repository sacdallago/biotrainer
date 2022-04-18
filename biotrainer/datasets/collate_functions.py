import torch

from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence


def pad_sequences(batch, padding_value=-100, batch_first=True):
    # batch is a list of the samples returned by your __get_item__ method in your CustomDataset
    seq_ids, X, Y = zip(*batch)
    X = pad_sequence(X, batch_first=batch_first, padding_value=padding_value)
    Y = pad_sequence(Y, batch_first=batch_first, padding_value=padding_value)
    return list(seq_ids), X, Y


# TODO: use this padding function instead of above
def pad_embeddings(batch: List[Tuple[torch.Tensor, torch.Tensor]], padding_value=-100, batch_first=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """

    :param batch: An iteration batch from the ResidueEmbeddingsDataset class [(ID, emnbedding, label)]
    :return: Returns padded embeddings, labels, lengths and True/False mask for padding of input embeddings.
    """

    # Each element in "batch" is a tuple [(ID, emnbedding, label)]

    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)

    # Sequence ids:
    seq_ids = [x[0] for x in sorted_batch]

    # Get each residue-embedding and pad it
    sequences = [x[1] for x in sorted_batch]
    padded_sequences = pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)

    # Also need to store the length of each sequence
    # This is later needed in order to unpad the sequences
    lengths = torch.LongTensor([len(x) for x in sequences])

    # Don't forget to grab the labels of the *sorted* batch
    labels_padded = pad_sequence([x[2] for x in sorted_batch], batch_first=batch_first, padding_value=padding_value)

    # Turn the lengths into Boolean masks
    masks = torch.arange(lengths.max())[None, :] < lengths[:, None]  # [batchsize, seq_len]

    # Premute the embeddings to fit the LA architecture
    return seq_ids, padded_sequences, labels_padded, lengths, masks
