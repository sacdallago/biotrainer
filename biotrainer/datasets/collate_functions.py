from torch.nn.utils.rnn import pad_sequence


def pad_sequences(batch, padding_value=-100, batch_first=True):
    # batch is a list of the samples returned by your __get_item__ method in your CustomDataset
    seq_ids, X, Y = zip(*batch)
    X = pad_sequence(X, batch_first=batch_first, padding_value=padding_value)
    Y = pad_sequence(Y, batch_first=batch_first, padding_value=padding_value)
    return (list(seq_ids), X, Y)
