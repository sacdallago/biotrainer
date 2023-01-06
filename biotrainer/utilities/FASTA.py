import re
import logging

from Bio import SeqIO
from typing import Dict, List
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

        result[interaction_id] = attribute_dict

    return result


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