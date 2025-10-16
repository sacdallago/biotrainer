from typing import List, Dict, Tuple

from .biotrainer_seq_record import BiotrainerSequenceRecord

from ..utilities.constants import INTERACTION_INDICATOR


def merge_protein_interactions(seq_records: List[BiotrainerSequenceRecord]) -> Dict[str, Dict[str, str]]:
    """
    :param seq_records: a list of BiotrainerSequenceRecord
    :return: A dictionary of ids and their attributes
    """
    result = {}

    seq_id_to_hash = {seq_record.seq_id: seq_record.get_hash() for seq_record in seq_records}

    for seq_record in seq_records:
        seq_id = seq_record.seq_id
        seq_hash = seq_record.get_hash()
        interactor = seq_record.get_ppi()
        interactor_hash = seq_id_to_hash[interactor]
        if not interactor:
            raise ValueError(f"Sequence {seq_id} does not have a valid interactor!")

        interaction_id = f"{seq_hash}{INTERACTION_INDICATOR}{interactor_hash}"
        interaction_id_flipped = f"{interactor_hash}{INTERACTION_INDICATOR}{seq_hash}"

        # Check that target annotations and sets are consistent:
        for int_id in [interaction_id, interaction_id_flipped]:
            if int_id in result.keys():
                if seq_record.get_target() != result[int_id]["TARGET"]:
                    raise ValueError(f"Interaction multiple times present in fasta file, but TARGET=values are "
                                     f"different for {int_id}!")
                if seq_record.get_set() != result[int_id]["SET"]:
                    raise ValueError(f"Interaction multiple times present in fasta file, but SET=sets are "
                                     f"different for {int_id}!")

        if interaction_id_flipped not in result.keys():
            result[interaction_id] = seq_record.attributes

    return result


def get_split_lists(id2sets: dict) -> Tuple[List[str], List[str], Dict[str, List[str]], List[str]]:
    """ Parse splits from input_file and check that all set annotations are given and correct """
    training_ids = []
    validation_ids = []
    testing_ids = {}
    prediction_ids = []

    incorrect_sets_exception = lambda idx, split: ValueError(f"FASTA header must contain SET. "
                                                             f"Id: {idx}; SET={split}")

    for idx, split in id2sets.items():
        split = split.lower() if split else ""
        if split == "":
            raise incorrect_sets_exception(idx, split)
        match split:
            case "train":
                training_ids.append(idx)
            case "val":
                validation_ids.append(idx)
            case "pred":
                prediction_ids.append(idx)
            case _:  # Treat all other sets as testing sets
                if split not in testing_ids:
                    testing_ids[split] = []
                testing_ids[split].append(idx)

    if len(training_ids) == 0:
        raise ValueError("No training sequences found in input file!")
    if len(testing_ids) == 0:
        raise ValueError("No test sets and sequences found in input file!")

    return training_ids, validation_ids, testing_ids, prediction_ids
