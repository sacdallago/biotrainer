import re
import logging

from typing import Dict, List, Tuple
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)


def attributes_from_seqrecords(sequences: List[SeqRecord]) -> Dict[str, Dict[str, str]]:
    """
    :param sequences: a list of SeqRecords
    :return: A dictionary of ids and their attributes
    """

    result = dict()

    for sequence in sequences:
        result[sequence.id] = {key: value for key, value in re.findall(r"([A-Z_]+)=([A-z0-9]+)", sequence.description)}

    return result


def _get_split_lists(id2attributes: dict) -> Tuple[list, list, list]:
    training_ids = list()
    validation_ids = list()
    testing_ids = list()

    # Sanity check: labels must contain SET and VALIDATION attributes
    for idx in id2attributes.keys():
        split = id2attributes[idx].get("SET")

        if split == 'train':
            val = id2attributes[idx].get("VALIDATION")

            try:
                val = eval(val)
            except NameError:
                pass

            if val is True:
                validation_ids.append(idx)
            elif val is False:
                training_ids.append(idx)
            else:
                Exception(
                    f"Sample in SET train must contain VALIDATION attribute. "
                    f"Validation must be True or False. "
                    f"Id: {idx}; VALIDATION={val}")

        elif split == 'test':
            testing_ids.append(idx)
        else:
            Exception(f"Labels FASTA header must contain SET. SET must be either 'train' or 'test'. "
                      f"Id: {idx}; SET={split}")

    return training_ids, validation_ids, testing_ids


def get_sets_from_labels(label_sequences: List[SeqRecord]) -> Tuple[List[str], List[str], List[str]]:
    id2attributes = attributes_from_seqrecords(label_sequences)

    training_ids, validation_ids, testing_ids = _get_split_lists(id2attributes)

    return training_ids, validation_ids, testing_ids


def get_sets_from_single_fasta(sequences_file: List[SeqRecord]) -> Tuple[Dict, List[str], List[str], List[str]]:
    id2attributes = attributes_from_seqrecords(sequences_file)

    seq_labels = {seq_id: seq_vals["TARGET"] for seq_id, seq_vals in id2attributes.items()}
    training_ids, validation_ids, testing_ids = _get_split_lists(id2attributes)

    return seq_labels, training_ids, validation_ids, testing_ids


def read_FASTA(path: str) -> List[SeqRecord]:
    """
    Helper function to read FASTA file.

    :param path: path to a valid FASTA file
    :return: a list of SeqRecord objects.
    """
    try:
        return list(SeqIO.parse(path, "fasta"))
    except FileNotFoundError:
        raise  # Already says "No such file or directory"
    except Exception as e:
        raise ValueError(f"Could not parse '{path}'. Are you sure this is a valid fasta file?") from e