import logging

from typing import Tuple, List, Optional, Dict

logger = logging.getLogger(__name__)


def get_split_lists(id2attributes: dict) -> Tuple[List[str], List[str], List[str]]:
    training_ids = list()
    validation_ids = list()
    testing_ids = list()

    # Sanity check: labels must contain SET and VALIDATION attributes
    for idx in id2attributes.keys():
        split = id2attributes[idx].get("SET")

        if split == 'train':
            val = id2attributes[idx].get("VALIDATION")

            try:
                val = eval(val.capitalize())
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


def revert_mappings(protocol: str, test_predictions: List, class_int2str: Optional[Dict[int, str]] = None):
    # If residue-to-class problem, map the integers back to the class labels (single letters)
    if protocol == 'residue_to_class':
        return ["".join(
            [class_int2str[p] for p in prediction]
        ) for prediction in test_predictions]

    # If sequence/residues-to-class problem, map the integers back to the class labels (whatever length)
    elif protocol == "sequence_to_class" or protocol == "residues_to_class":
        return [class_int2str[p] for p in test_predictions]
    else:
        return test_predictions
