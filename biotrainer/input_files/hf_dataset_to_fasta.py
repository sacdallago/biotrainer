from pathlib import Path
from datasets import load_dataset
from typing import Any, Dict, List, Optional, Tuple

from ..protocols import Protocol
from ..utilities import get_logger

logger = get_logger(__name__)


def _write_hf_to_fasta(
        sequences: List[str],
        targets: List[Any],
        set_values: List[str],
        hf_storage_path: Path,
        masks: Optional[List[Any]] = None, ) -> Optional[Path]:
    """
    Converts sequences, targets, and optional masks from a HuggingFace dataset into FASTA file(s).

    Args:
        sequences (List[str]): A list of protein sequences.
        targets (List[Any]): A list of target values.
        set_values (List[str]): A list of SET values corresponding to each sequence.
        hf_storage_path (Path): Path to the HuggingFace storage path.
        masks (Optional[List[Any]]): A list of mask values, or None if masks are not provided.

    Raises:
        ValueError: If the lengths of sequences, targets, masks (if masks are provided), and set_values do not match.
        IOError: If there is an issue writing to the output files.
    """
    if not masks:
        if not (len(sequences) == len(targets) == len(set_values)):
            raise ValueError("The number of sequences, targets, and set_values must be the same.")
    else:
        if not (len(sequences) == len(targets) == len(set_values) == len(masks)):
            raise ValueError("The number of sequences, targets, set_values, and masks must be the same.")

    input_file_name = hf_storage_path / "input.fasta"
    with open(input_file_name, "w") as input_file:
        # Write the data
        for idx in range(len(sequences)):
            seq_id = f"Seq{idx + 1}"
            seq = sequences[idx]
            target = targets[idx]
            set_val = set_values[idx]
            mask = masks[idx] if masks is not None else None

            seq_header = f">{seq_id} TARGET={target} SET={set_val}"
            if mask is not None:
                seq_header += f" MASK={mask}"
            input_file.write(f"{seq_header}\n{seq}\n")

        return input_file_name


def _process_subset(
        current_subset: Any,
        sequence_column: str,
        target_column: str,
        mask_column: Optional[str] = None
) -> Tuple[List[str], List[Any], Optional[List[Any]]]:
    """
    Processes a single subset, verifying the presence of required columns
    and extracting sequences, targets, and optionally masks.

    Args:
        current_subset (Any): The subset to process. This is typically a HuggingFace dataset or similar structure.
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
    _verify_dataset_column(current_subset, sequence_column)
    _verify_dataset_column(current_subset, target_column)
    if mask_column:
        _verify_dataset_column(current_subset, mask_column)

    # Extract data
    sequences = current_subset[sequence_column]
    targets = current_subset[target_column]
    masks = current_subset[mask_column] if mask_column else None

    return sequences, targets, masks


def _determine_set_name(subset_name: str) -> str:
    """
    Determines the corresponding set name ("TRAIN", "VAL", "TEST", "PRED") based on the provided subset name.

    This function normalizes and categorizes the input subset name into one of the standard set names. If the input does not
    match any of the expected patterns ("train", "val", "test", "pred"),
    it logs a warning and assigns the subset to "TEST-{subset_name}" by default.

    Args:
        subset_name (str): The name of the subset (e.g., "train1", "validation", "testing").

    Returns:
        str: The normalized set name ("TRAIN", "VAL", "TEST", "PRED", "TEST-{subset_name}").

    Logs:
        Warning: Logs a warning if the subset name is unrecognized and defaults to "TEST".

    """
    lower_subset_name = subset_name.lower()
    if lower_subset_name.startswith("train"):
        return "TRAIN"
    elif lower_subset_name.startswith("val"):
        return "VAL"
    elif lower_subset_name.startswith("test"):
        return "TEST"
    elif lower_subset_name.startswith("pred"):
        return "PRED"
    else:
        logger.warning(f"Unrecognized subset name '{subset_name}'. Assigning to 'TEST-{lower_subset_name}'.")
        return f"TEST-{lower_subset_name}"


def _verify_dataset_column(dataset: Any, column: str) -> None:
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


def _load_and_split_hf_dataset(hf_map: Dict) -> Tuple[List[str], List[Any], List[Any], List[str]]:
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
    path = hf_map["path"]
    sequence_column = hf_map["sequence_column"]
    target_column = hf_map["target_column"]
    mask_column = hf_map["mask_column"] if "mask_column" in hf_map else None
    subset_name = hf_map["subset"] if "subset" in hf_map else None

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

    # Collect all available subsets
    subset_names = list(dataset.keys())
    if not subset_names:
        raise Exception(f"No subsets found in the dataset at path '{path}'.")

    sequences = []
    targets = []
    masks = []
    set_values = []

    if len(subset_names) >= 3:
        logger.info(f"{len(subset_names)} subsets found: {subset_names}. Processing each separately.")
        for subset_name in subset_names:
            current_subset = dataset[subset_name]
            if current_subset is None:
                logger.warning(f"Subset '{subset_name}' is None. Skipping.")
                continue

            # Process subset
            current_sequences, current_targets, current_masks = _process_subset(current_subset, sequence_column,
                                                                                target_column, mask_column)

            # Determine SET value based on split name
            current_set_value = _determine_set_name(subset_name)

            sequences.extend(current_sequences)
            targets.extend(current_targets)
            set_values.extend([current_set_value] * len(current_sequences))
            if current_masks is not None:
                masks.extend(current_masks)
            else:
                masks.extend([None] * len(current_sequences))

    else:
        raise ValueError(
            f"Expected at least 3 subsets (TRAIN, VAL, TEST) in the dataset at path '{path}'. Found: {subset_names}."
        )

    return sequences, targets, masks, set_values


def process_hf_dataset_to_fasta(
        hf_storage_path: Path,
        hf_map: Dict) -> Optional[Path]:
    """
    Loads a HuggingFace dataset, splits it according to the protocol, and writes the data to FASTA files.

    Args:
        hf_storage_path (Path): Path to the HuggingFace storage directory.
        hf_map (Dict): A mapping of HuggingFace dataset option names to their respective ConfigOption instances.

    Raises:
        Exception: If there is an issue during the creation of the required files or processing the dataset.
    """
    try:
        sequences, targets, masks, set_values = _load_and_split_hf_dataset(hf_map)
    except Exception as e:
        raise Exception(f"Failed to load and split HuggingFace dataset: {e}")

    try:
        input_file_path = _write_hf_to_fasta(
            sequences=sequences,
            targets=targets,
            masks=masks if any([m is not None for m in masks]) else None,
            set_values=set_values,
            hf_storage_path=hf_storage_path,
        )

        logger.info("HuggingFace dataset downloaded and processed successfully.")
        return input_file_path
    except Exception as e:
        raise Exception(f"Failed to write huggingface dataset to FASTA files: {e}")
