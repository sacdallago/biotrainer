import re

from typing import Iterable, List, Optional


def preprocess_sequences_with_whitespaces(sequences: Iterable[str], mask_token: Optional[str]) -> List[str]:
    # Remove rare amino acids
    sequences_cleaned = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    # Transformers need spaces between the amino acids

    def add_whitespaces(sequence: str) -> str:
        """ Do not split mask token """
        result = ""
        sequence_parts = sequence.split(sep=mask_token)
        for sequence_part in sequence_parts:
            if len(result) > 0:
                result += " "

            result += " ".join(list(sequence_part))
            result += f" {mask_token}"
        result = result[:-(len(mask_token)+1)].strip()  # +1 for last whitespace
        return result

    if mask_token is not None:
        sequences_with_spaces = [add_whitespaces(sequence) for sequence in sequences_cleaned]
    else:
        sequences_with_spaces = [" ".join(list(sequence)) for sequence in sequences_cleaned]
    return sequences_with_spaces


def preprocess_sequences_without_whitespaces(sequences: Iterable[str], mask_token: Optional[str]) -> List[str]:
    # Remove rare amino acids
    sequences_cleaned = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    return sequences_cleaned


def preprocess_sequences_for_prostt5(sequences: Iterable[str], mask_token: Optional[str]) -> List[str]:
    preprocessed_sequences = preprocess_sequences_with_whitespaces(sequences, mask_token=mask_token)

    # We have AAs, so we need to tell the model we want to translate AA->3Di.
    # To do so, we prepend "<AA2fold>" while keeping residues in upper-case.
    prefixed_sequences = [
        "<AA2fold>" + " " + seq.upper() for seq in preprocessed_sequences
    ]
    return prefixed_sequences
