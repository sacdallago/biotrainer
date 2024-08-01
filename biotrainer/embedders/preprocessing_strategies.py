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
