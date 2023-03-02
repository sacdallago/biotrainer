import logging

from typing import Dict

logger = logging.getLogger(__name__)


def revert_mappings(protocol: str, test_predictions: Dict, class_int2str: Dict[int, str]) -> Dict[str, str]:
    # If residue-to-class problem, map the integers back to the class labels (single letters)
    if protocol == 'residue_to_class':
        return {seq_id: "".join([class_int2str[p] for p in prediction])
                for seq_id, prediction in test_predictions.items()}

    # If sequence/residues-to-class problem, map the integers back to the class labels (whatever length)
    elif protocol == "sequence_to_class" or protocol == "residues_to_class":
        return {seq_id: class_int2str[prediction] for seq_id, prediction in test_predictions.items()}
    else:
        return test_predictions
