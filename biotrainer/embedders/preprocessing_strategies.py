import re

from typing import Iterable, List


def preprocess_sequences_with_whitespaces(sequences: Iterable[str]) -> List[str]:
    # Remove rare amino acids
    sequences_cleaned = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    # Transformers need spaces between the amino acids
    sequences_with_spaces = [" ".join(list(sequence)) for sequence in sequences_cleaned]
    return sequences_with_spaces


def preprocess_sequences_without_whitespaces(sequences: Iterable[str]) -> List[str]:
    # Remove rare amino acids
    sequences_cleaned = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    return sequences_cleaned


def preprocess_sequences_for_prostt5(sequences: Iterable[str]) -> List[str]:
    preprocessed_sequences = preprocess_sequences_with_whitespaces(sequences)

    # We have AAs, so we need to tell the model we want to translate AA->3Di.
    # To do so, we prepend "<AA2fold>" while keeping residues in upper-case.
    prefixed_sequences = [
        "<AA2fold>" + " " + seq.upper() for seq in preprocessed_sequences
    ]
    return prefixed_sequences
