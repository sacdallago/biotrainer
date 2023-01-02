import logging

from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)


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
