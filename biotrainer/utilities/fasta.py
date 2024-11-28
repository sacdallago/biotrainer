import re
import logging

from Bio import SeqIO
from typing import Dict, List, Tuple, Any, Optional
from Bio.SeqRecord import SeqRecord

from ..utilities import INTERACTION_INDICATOR

logger = logging.getLogger(__name__)


def get_attributes_from_seqrecords(sequences: List[SeqRecord]) -> Dict[str, Dict[str, str]]:
    """
    :param sequences: a list of SeqRecords
    :return: A dictionary of ids and their attributes
    """

    result = dict()

    for sequence in sequences:
        result[sequence.id] = {key: value for key, value in re.findall(r"([A-Z_]+)=(-?[A-z0-9]+-?[A-z0-9]*[.0-9]*)",
                                                                       sequence.description)}

    return result


def get_attributes_from_seqrecords_for_protein_interactions(sequences: List[SeqRecord]) -> Dict[str, Dict[str, str]]:
    """
    :param sequences: a list of SeqRecords
    :return: A dictionary of ids and their attributes
    """

    result = dict()

    for sequence in sequences:
        attribute_dict = {key: value for key, value
                          in re.findall(r"([A-Z_]+)=(-?[A-z0-9]+-?[A-z0-9]*[.0-9]*)", sequence.description)}
        interaction_id = f"{sequence.id}{INTERACTION_INDICATOR}{attribute_dict['INTERACTOR']}"
        interaction_id_flipped = f"{attribute_dict['INTERACTOR']}{INTERACTION_INDICATOR}{sequence.id}"

        # Check that target annotations and sets are consistent:
        for int_id in [interaction_id, interaction_id_flipped]:
            if int_id in result.keys():
                if attribute_dict["TARGET"] != result[int_id]["TARGET"]:
                    raise Exception(f"Interaction multiple times present in fasta file, but TARGET=values are "
                                    f"different for {int_id}!")
                if attribute_dict["SET"] != result[int_id]["SET"]:
                    raise Exception(f"Interaction multiple times present in fasta file, but SET=sets are "
                                    f"different for {int_id}!")

        if interaction_id_flipped not in result.keys():
            result[interaction_id] = attribute_dict

    return result


def get_split_lists(id2attributes: dict) -> Tuple[List[str], List[str], List[str]]:
    training_ids = list()
    validation_ids = list()
    testing_ids = list()

    # Check that all set annotations are given and correct
    for idx, attributes in id2attributes.items():
        split = attributes.get("SET")
        if split is None or split.lower() not in ["train", "val", "test"]:
            raise Exception(f"Labels FASTA header must contain SET. SET must be either 'train', 'val' or 'test'. "
                            f"Id: {idx}; SET={split}")
    # Decide between old VALIDATION=True/False annotation and new split (train/val/test) annotation
    validation_annotation: bool = any(
        [attributes.get("VALIDATION") is not None for attributes in id2attributes.values()])
    set_annotation: bool = any([attributes.get("SET").lower() == "val" for attributes in id2attributes.values()])
    deprecation_exception = Exception(f"Split annotations are given with both, VALIDATION=True/False and SET=val.\n"
                                      f"This is redundant, please prefer to only use SET=train/val/test "
                                      f"for all sequences!\n"
                                      f"VALIDATION=True/False is deprecated and only supported if no SET=val "
                                      f"annotation is present in the dataset.")
    if validation_annotation and set_annotation:
        raise deprecation_exception
    for idx, attributes in id2attributes.items():
        split = attributes.get("SET").lower()

        if split == 'train':
            # Annotation VALIDATION=True/False (DEPRECATED for biotrainer > 0.2.1)
            if validation_annotation:
                val = id2attributes[idx].get("VALIDATION")

                try:
                    val = eval(val.capitalize())
                    if val is True:
                        validation_ids.append(idx)
                    elif val is False:
                        training_ids.append(idx)
                except AttributeError:
                    raise Exception(
                        f"Sample in SET train must contain VALIDATION attribute. \n"
                        f"Validation must be True or False. \n"
                        f"Id: {idx}; VALIDATION={val} \n"
                        f"Alternatively, split annotations can be given via "
                        f"SET=train/val/test (for biotrainer > 0.2.1)")
            else:
                training_ids.append(idx)

        elif split == 'val':
            if validation_annotation or not set_annotation:
                raise deprecation_exception
            validation_ids.append(idx)
        elif split == 'test':
            testing_ids.append(idx)
        else:
            raise Exception(f"Labels FASTA header must contain SET. SET must be either 'train', 'val' or 'test'. "
                            f"Id: {idx}; SET={split}")

    return training_ids, validation_ids, testing_ids


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


def hf_to_fasta(
    sequences: List[str],
    targets: List[Any],
    set_values: List[str],
    sequences_file_name: str,
    targets_file_name: Optional[str] = None
) -> None:
    """
    Converts sequences and targets from a HuggingFace dataset into FASTA file(s).

    Args:
        sequences (List[str]): A list of protein sequences.
        targets (List[Any]): A list of target values.
        set_values (List[str]): A list of SET values corresponding to each sequence.
        sequences_file_name (str): Path and filename for the output sequence FASTA file.
        targets_file_name (Optional[str], optional): Path and filename for the output target FASTA file.
            Defaults to None.

    Raises:
        ValueError: If the lengths of sequences, targets, and set_values do not match.
        IOError: If there is an issue writing to the output files.
    """
    if not (len(sequences) == len(targets) == len(set_values)):
        raise ValueError("The number of sequences, targets, and set_values must be the same.")

    try:
        if targets_file_name is None:
            # Single FASTA file mode
            with open(sequences_file_name, 'w') as seq_file:
                for idx, (seq, target, set_val) in enumerate(zip(sequences, targets, set_values), start=1):
                    seq_id = f"Seq{idx}"
                    header = f">{seq_id} SET={set_val} TARGET={target}"
                    seq_file.write(f"{header}\n{seq}\n")
        else:
            # Separate FASTA files mode
            with open(sequences_file_name, 'w') as seq_file, open(targets_file_name, 'w') as tgt_file:
                for idx, (seq, target, set_val) in enumerate(zip(sequences, targets, set_values), start=1):
                    seq_id = f"Seq{idx}"

                    # Write to sequence FASTA file
                    seq_header = f">{seq_id} SET={set_val}"
                    seq_file.write(f"{seq_header}\n{seq}\n")

                    # Write to target FASTA file
                    tgt_header = f">{seq_id} SET={set_val}"
                    tgt_file.write(f"{tgt_header}\n{target}\n")

    except IOError as e:
        raise IOError(f"Error writing to FASTA file(s): {e}")
