import re
import logging

from Bio import SeqIO
from typing import Dict, List, Tuple, Union, Any, Optional
from Bio.SeqRecord import SeqRecord
from datasets import load_dataset, concatenate_datasets

from ..utilities import INTERACTION_INDICATOR
from ..protocols import Protocol

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
    labels_file_name: Optional[str] = None,
    masks: Optional[List[Any]] = None,
    masks_file_name: Optional[str] = None
) -> None:
    """
    Converts sequences, targets, and optional masks from a HuggingFace dataset into FASTA file(s).

    Args:
        sequences (List[str]): A list of protein sequences.
        targets (List[Any]): A list of target values.
        set_values (List[str]): A list of SET values corresponding to each sequence.
        sequences_file_name (str): Path and filename for the output sequence FASTA file.
        labels_file_name (Optional[str], optional): Path and filename for the output target FASTA file.
            Defaults to None.
        masks (Optional[List[Any]]): A list of mask values, or None if masks are not provided.
        masks_file_name (Optional[str], optional): Path and filename for the output mask FASTA file.
            Defaults to None.

    Raises:
        ValueError: If the lengths of sequences, targets, masks (if masks are provided), and set_values do not match.
        IOError: If there is an issue writing to the output files.
    """
    if not masks_file_name:
        if not (len(sequences) == len(targets) == len(set_values)):
            raise ValueError("The number of sequences, targets, and set_values must be the same.")
    else:
        if not (len(sequences) == len(targets) == len(set_values) == len(masks)):
            raise ValueError("The number of sequences, targets, set_values, and masks must be the same.")

    files = {}
    try:
        # Open the necessary files
        seq_file = open(sequences_file_name, 'w')
        files['seq_file'] = seq_file

        if labels_file_name:
            tgt_file = open(labels_file_name, 'w')
            files['tgt_file'] = tgt_file
        else:
            tgt_file = None

        if masks_file_name:
            mask_file = open(masks_file_name, 'w')
            files['mask_file'] = mask_file
        else:
            mask_file = None

        # Write the data
        for idx in range(len(sequences)):
            seq_id = f"Seq{idx+1}"
            seq = sequences[idx]
            target = targets[idx]
            set_val = set_values[idx]
            mask = masks[idx] if masks is not None else None

            # Write to sequence file
            if tgt_file is None:
                # Include target in the header
                seq_header = f">{seq_id} SET={set_val} TARGET={target}"
            else:
                seq_header = f">{seq_id}"
            seq_file.write(f"{seq_header}\n{seq}\n")

            if tgt_file is not None:
                # Write to target file
                tgt_header = f">{seq_id} SET={set_val}"
                tgt_file.write(f"{tgt_header}\n{target}\n")

            if mask_file is not None:
                # Write to mask file
                mask_header = f">{seq_id}"
                mask_file.write(f"{mask_header}\n{mask}\n")

        # Close all files
        for f in files.values():
            f.close()

    except IOError as e:
        # Ensure all files are closed in case of exception
        for f in files.values():
            f.close()
        raise IOError(f"Error writing to FASTA file(s): {e}")

def process_split(
    split_dataset: Any,
    sequence_column: str,
    target_column: str,
    mask_column: Optional[str] = None
) -> Tuple[List[str], List[Any], Optional[List[Any]]]:
    """
    Processes a single dataset split, verifying the presence of required columns
    and extracting sequences, targets, and optionally masks.

    Args:
        split_dataset (Any): The dataset split to process. This is typically a HuggingFace dataset or similar structure.
        sequence_column (str): The name of the column containing the sequences.
        target_column (str): The name of the column containing the target values.
        mask_column (Optional[str]): The name of the column containing mask values, if applicable. Defaults to None.

    Returns:
        Tuple[List[str], List[Any], Optional[List[Any]]]: A tuple containing:
            - List[str]: A list of sequences.
            - List[Any]: A list of target values.
            - Optional[List[Any]]: A list of mask values, or None if no mask column is provided.

    Raises:
        Exception: If any of the specified columns are missing from the dataset.
    """
    # Verify columns
    verify_column(split_dataset, sequence_column)
    verify_column(split_dataset, target_column)
    if mask_column:
        verify_column(split_dataset, mask_column)

    # Extract data
    sequences = split_dataset[sequence_column]
    targets = split_dataset[target_column]
    masks = split_dataset[mask_column] if mask_column else None

    return sequences, targets, masks

def determine_set_name(split_name: str) -> str:
    """
    Determines the corresponding set name ("TRAIN", "VAL", "TEST") based on the provided split name.

    This function normalizes and categorizes the input split name into one of the standard set names. If the input does not
    match any of the expected patterns ("train", "val", "test"), it logs a warning and assigns the split to "TEST" by default.

    Args:
        split_name (str): The name of the split (e.g., "train1", "validation", "testing").

    Returns:
        str: The normalized set name ("TRAIN", "VAL", "TEST").

    Logs:
        Warning: Logs a warning if the split name is unrecognized and defaults to "TEST".

    """
    lower_split = split_name.lower()
    if lower_split.startswith("train"):
        return "TRAIN"
    elif lower_split.startswith("val"):
        return "VAL"
    elif lower_split.startswith("test"):
        return "TEST"
    else:
        logger.warning(f"Unrecognized split name '{split_name}'. Assigning to 'TEST'.")
        return "TEST"

def verify_column(dataset: Any, column: str) -> None:
    """
    Verifies that the specified column exists in the given dataset.

    Args:
        dataset (Any): The dataset to verify. This is expected to have a `column_names` attribute,
                       such as a HuggingFace dataset or a similar structure.
        column (str): The name of the column to check for existence.

    Raises:
        Exception: If the specified column is not found in the dataset.
    """
    if column not in dataset.column_names:
        raise Exception(
            f"Column '{column}' not found in the dataset."
        )

def load_and_split_hf_dataset(hf_map: Dict) -> Tuple[List[str], List[Any], List[Any], List[str]]:
    """
    Loads a HuggingFace dataset and splits it into sequences, targets, masks (if available), and set values.

    Args:
        hf_map (Dict[str, Any]): A mapping of configuration options for loading the HuggingFace dataset.
                                 Expected keys include:
                                 - "path" (str): The dataset path.
                                 - "sequence_column" (str): Name of the sequence column.
                                 - "target_column" (str): Name of the target column.
                                 - "mask_column" (Optional[str]): Name of the mask column (if applicable).
                                 - "subset" (Optional[str]): Subset name, if required.

    Returns:
        Tuple[List[str], List[Any], Optional[List[Any]], List[str]]:
            - List[str]: Sequences extracted from the dataset.
            - List[Any]: targets corresponding to the sequences.
            - Optional[List[Any]]: Masks, if available; otherwise, None.
            - List[str]: Set values (e.g., "TRAIN", "VAL", "TEST") for each sequence.

    Raises:
        ValueError: If loading the dataset fails due to missing or invalid configuration.
        Exception: If the required splits or columns are missing, or if other errors occur during processing.

    """
    # Extract parameters from hf_map
    path = hf_map["path"].value
    sequence_column = hf_map["sequence_column"].value
    target_column = hf_map["target_column"].value
    mask_column = hf_map["mask_column"].value if "mask_column" in hf_map else None
    subset_name = hf_map["subset"].value if "subset" in hf_map else None

    logger.info(f"Loading HuggingFace dataset from path: {path}")

    # Load dataset
    try:
        dataset = load_dataset(path, subset_name if subset_name is not None else 'default')
    except ValueError as e:
        error_msg = ("Loading the dataset from Hugging Face failed. "
                     "If the dataset requires a 'subset', you can specify it using the 'subset' option in the config file.\n"
                     f"Error: {e}")
        raise ValueError(error_msg)
    except Exception as e:
        raise Exception(f"Loading the dataset from Hugging Face failed. Error: {e}")

    # Collect all available splits
    available_splits = list(dataset.keys())
    if not available_splits:
        raise Exception(f"No splits found in the dataset at path '{path}'.")

    sequences = []
    targets = []
    masks = []
    set_values = []

    if len(available_splits) == 3:
        logger.info(f"Three splits found: {available_splits}. Processing each split separately.")
        for split_name in available_splits:
            split_dataset = dataset[split_name]
            if split_dataset is None:
                logger.warning(f"Split '{split_name}' is None. Skipping.")
                continue

            # Process split
            split_sequences, split_targets, split_masks = process_split(split_dataset, sequence_column,
                                                                        target_column, mask_column)

            # Determine SET value based on split name
            set_name = determine_set_name(split_name)

            sequences.extend(split_sequences)
            targets.extend(split_targets)
            set_values.extend([set_name] * len(split_sequences))
            if split_masks is not None:
                masks.extend(split_masks)
            else:
                masks.extend([None] * len(split_sequences))

    else:
        raise Exception(
            f"Expected 3 splits (TRAIN, VAL, TEST) in the dataset at path '{path}'. Found {len(available_splits)}."
        )

    return sequences, targets, masks, set_values
