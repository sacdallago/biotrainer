import itertools
import logging
import torch

from collections import Counter
from typing import Tuple, List, Dict


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


def get_class_weights(
        id2target: Dict[str, int], class_str2int: Dict[str, int], class_int2str: Dict[int, str]
) -> torch.FloatTensor:
    # concatenate all labels irrespective of protein to count class sizes
    counter = Counter(list(itertools.chain.from_iterable(
        [list(labels) for labels in id2target.values()]
    )))
    # total number of samples in the set irrespective of classes
    n_samples = sum([counter[idx] for idx in range(len(class_str2int))])
    # balanced class weighting (inversely proportional to class size)
    class_weights = [
        (n_samples / (len(class_str2int) * counter[idx])) for idx in range(len(class_str2int))
    ]

    logger.info(f"Total number of sequences/residues: {n_samples}")
    logger.info("Individual class counts and weights:")
    for c in counter:
        logger.info(f"\t{class_int2str[c]} : {counter[c]} ({class_weights[c]:.3f})")
    return torch.FloatTensor(class_weights)
