# Script to convert FLIP datasets to autoeval biotrainer standard
import os
import shutil
import tempfile

from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

from .flip_datasets import FLIP_DATASETS

from ...protocols import Protocol
from ...input_files import read_FASTA, write_FASTA, convert_deprecated_fastas

IGNORE_SPLITS = ["mixed_vs_human_2"]


def get_sequence_file_path(dataset_dir: Path, name: str) -> Path:
    """Get the appropriate sequence file path (preprocessed if available)"""
    raw_path = Path(f"{name}.fasta")
    preprocessed_path = Path("preprocessed") / raw_path

    if (dataset_dir / preprocessed_path).exists():
        return preprocessed_path
    return raw_path


def get_duplicates(split_name: str, all_seq_records, report: bool):
    seq_counts = {}
    for seq_record in all_seq_records:
        seq = seq_record.seq
        if seq not in seq_counts:
            seq_counts[seq] = []
        seq_counts[seq].append(seq_record)

    csv_report = ""
    duplicates = set()
    for seq, seq_records in seq_counts.items():
        count = len(seq_records)
        if count > 1:
            for seq_record in seq_records:
                duplicates.add(seq_record)

            targets = {seq.get_target() for seq in seq_records}
            target_string = ';'.join(targets) if all([target is not None for target in targets]) else "N/A"
            csv_line = (f"{split_name},{count},{';'.join([seq.seq_id for seq in seq_records])},"
                        f"{target_string}")
            csv_report += csv_line
            csv_report += "\n"
            print(csv_line)

    header = ""
    if not os.path.exists("flip_duplicates.csv"):
        header = "SPLIT,COUNT,IDS,TARGETS\n"

    if report:
        with open("flip_duplicates.csv", "a") as csv_file:
            if len(header) > 0:
                csv_file.write(header)
            csv_file.write(csv_report)

    return duplicates


def ensure_preprocessed_file(dataset_dir: Path, name: str, min_seq_length: int, max_seq_length: int) -> Path:
    """Ensure a preprocessed version of the file exists and return its path"""

    complete_name = dataset_dir.name.split('/')[-1] + '-' + name

    download_path = dataset_dir / f"{name}.fasta"
    preprocessed_path = dataset_dir / "preprocessed" / f"{name}.fasta"

    # If preprocessed file already exists, return its path
    if preprocessed_path.exists():
        return preprocessed_path

    # If raw file doesn't exist, we can't proceed
    if not download_path.exists():
        raise FileNotFoundError(f"Required file {download_path} not available for: {name}")

    # Preprocess the file
    all_seq_records = read_FASTA(str(download_path))

    keep_seqs = [
        seq_record for seq_record in all_seq_records
        if (min_seq_length <= len(seq_record.seq) <= max_seq_length and seq_record.get_set() != "nan")
    ]

    # Only keep unique sequences
    keep_seqs = {seq_record.get_hash(): seq_record for seq_record in keep_seqs}

    # Remove all duplicates
    duplicates = get_duplicates(complete_name, all_seq_records, False)
    duplicates_hashed = {dup.get_hash() for dup in duplicates}

    keep_seqs = {seq_ha: seq_record for seq_ha, seq_record in keep_seqs.items() if seq_ha not in duplicates_hashed}

    preprocessed_dir = dataset_dir / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True)

    n_written = write_FASTA(path=preprocessed_path, seq_records=list(keep_seqs.values()))
    assert n_written == len(keep_seqs)

    print(f"Preprocessed {download_path.name}: kept {n_written}/{len(all_seq_records)} sequences")
    return preprocessed_path


def preprocess(base_path: Path, min_seq_length: int, max_seq_length: int) -> None:
    """Preprocess all dataset files"""
    for dataset, dataset_info in tqdm(FLIP_DATASETS.items(), desc="Preprocessing datasets"):
        dataset_dir = base_path / dataset
        protocol = dataset_info["protocol"]
        if isinstance(protocol, str):
            protocol = Protocol.from_string(protocol)

        # Process all splits
        for split in dataset_info["splits"]:
            if split in IGNORE_SPLITS:
                continue

            try:
                if protocol in Protocol.per_sequence_protocols():
                    ensure_preprocessed_file(dataset_dir=dataset_dir, name=split,
                                             min_seq_length=min_seq_length,
                                             max_seq_length=max_seq_length)
                elif protocol in Protocol.per_residue_protocols():
                    ensure_preprocessed_file(dataset_dir, "sequences",
                                             min_seq_length=min_seq_length,
                                             max_seq_length=max_seq_length)
            except Exception as e:
                print(f"Error preprocessing {dataset}/{split}: {e}")

    print("FLIP data preprocessing completed!")


def get_dataset_dict(base_path: Path) -> Dict[str, Any]:
    """Build path dictionary for all FLIP datasets"""
    result_dict = {}

    for dataset, dataset_info in FLIP_DATASETS.items():
        result_dict[dataset] = {}
        dataset_dir = base_path / dataset
        protocol = dataset_info["protocol"]

        result_dict[dataset]["splits"] = []
        for split in dataset_info["splits"]:
            if split in IGNORE_SPLITS:
                continue

            split_data = {
                "name": split,
                "sequence_file": None,
                "labels_file": None,
                "mask_file": None
            }

            try:
                if protocol in Protocol.per_sequence_protocols():
                    sequence_file = get_sequence_file_path(dataset_dir,
                                                           split)
                    if (dataset_dir / sequence_file).exists():
                        split_data["sequence_file"] = str(dataset_dir / sequence_file)
                    else:
                        print(f"Missing sequence file for {dataset}/{split}")
                        continue

                elif protocol in Protocol.per_residue_protocols():
                    sequences_file = get_sequence_file_path(dataset_dir,
                                                            "sequences",
                                                            )

                    labels_file = Path(f"{split}.fasta")
                    mask_file = Path(f"resolved.fasta")

                    if (dataset_dir / sequences_file).exists() and (dataset_dir / labels_file).exists():
                        split_data["sequence_file"] = str(dataset_dir / sequences_file)
                        split_data["labels_file"] = str(dataset_dir / labels_file)
                    else:
                        print(f"Missing required files for {dataset}/{split}")
                        continue

                    if (dataset_dir / mask_file).exists():
                        split_data["mask_file"] = str(dataset_dir / mask_file)

                result_dict[dataset]["splits"].append(split_data)

            except Exception as e:
                print(f"Error processing {dataset}/{split}: {e}")

    return result_dict


def convert():
    print("Starting FLIP conversion...")
    # Example usage
    zip_path = Path("all_fastas.zip")  # Path to downloaded all_fastas.zip file
    result_path = Path("FLIP_converted")

    # Unpack zip
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Unpacking data archive..")
        shutil.unpack_archive(zip_path, tmp_dir)

        # Preprocess the datasets
        preprocess(Path(tmp_dir), min_seq_length=0, max_seq_length=1000000000000)

        # Get dataset dictionary
        dataset_dict = get_dataset_dict(Path(tmp_dir))

        # Merge to new biotrainer format
        print("Converting dataset...")
        for dataset, dataset_info in dataset_dict.items():
            dataset_path = f"{result_path}/{dataset}"
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path, exist_ok=True)

            for split in dataset_info["splits"]:
                name = split["name"]
                seq_file = split["sequence_file"]
                labels_file = split["labels_file"]
                mask_file = split["mask_file"]
                split_file_path = Path(f"{dataset_path}/{name}.fasta")
                convert_deprecated_fastas(result_file=str(split_file_path),
                                          sequence_file=seq_file,
                                          labels_file=labels_file,
                                          masks_file=mask_file,
                                          skip_sequence_on_failed_merge=True)

    print(f"FLIP data converted successfully, data is at {result_path}!")
