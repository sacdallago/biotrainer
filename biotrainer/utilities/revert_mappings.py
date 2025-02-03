from typing import Dict

from ..protocols import Protocol

def revert_mappings(protocol: Protocol, test_predictions: Dict, class_int2str: Dict[int, str]) -> Dict[str, str]:
    # If residue-to-class problem, map the integers back to the class labels (single letters)
    if protocol == Protocol.residue_to_class:
        return {seq_id: "".join([class_int2str[p] for p in prediction]
                                if type(prediction) == list else class_int2str[prediction])
                for seq_id, prediction in test_predictions.items()}

    # If sequence/residues-to-class problem, map the integers back to the class labels (whatever length)
    elif protocol == Protocol.sequence_to_class or protocol == Protocol.residues_to_class:
        return {seq_id: class_int2str[prediction] for seq_id, prediction in test_predictions.items()}
    else:
        return test_predictions
