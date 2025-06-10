import re

from pathlib import Path
from typing import Dict, Union, List

from .biotrainer_seq_record import BiotrainerSequenceRecord


def read_FASTA(path: Union[str, Path]) -> Dict[str, BiotrainerSequenceRecord]:
    """
    Pure Python FASTA file parser. Parses attributes on the go.

    :param path: path to a valid FASTA file
    :return: a dict of seq_id -> BiotrainerSequenceRecord objects
    """
    attributes_pattern = re.compile(r"([A-Z_]+)=(-?[A-z0-9]+-?[A-z0-9]*[.0-9]*)")

    records = {}
    try:
        with open(path, 'r') as file:
            current_id = ""
            current_attributes = {}
            current_seq = ""

            for line in file:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('>'):
                    # Save the previous sequence if it exists
                    if current_id:
                        if current_id in records:
                            raise ValueError(f"Duplicated sequence id in fasta file: {current_id}!")
                        records[current_id] = BiotrainerSequenceRecord(
                            seq_id=current_id,
                            attributes=current_attributes,
                            seq=current_seq
                        )

                    # Parse the header line
                    header = line[1:].strip()
                    parts = header.split(maxsplit=1)
                    current_id = parts[0]
                    current_description = parts[1] if len(parts) > 1 else ""
                    current_attributes = {key: value for key, value in attributes_pattern.findall(current_description)}
                    current_seq = ""
                else:
                    current_seq = line

            # Ensure last sequence is added
            if current_id:
                if current_id in records:
                    raise ValueError(f"Duplicated sequence id in fasta file: {current_id}!")
                records[current_id] = BiotrainerSequenceRecord(
                    seq_id=current_id,
                    attributes=current_attributes,
                    seq=current_seq
                )

        return records

    except FileNotFoundError as e:
        raise e
    except ValueError as e:
        raise e
    except Exception as e:
        raise ValueError(f"Could not parse '{path}'. Are you sure this is a valid fasta file?") from e


def write_FASTA(path: Union[str, Path], seq_records: List[BiotrainerSequenceRecord]) -> int:
    """
    Save a list of BiotrainerSequenceRecord objects to a FASTA file.

    :param path: Path to store the file
    :param seq_records: List of BiotrainerSequenceRecord objects
    :return: number of sequence records written (following biopython standard)
    """
    with open(path, 'w') as fasta_file:
        n_written = 0
        for seq_record in seq_records:
            attributes_str = " ".join([f"{key.upper()}={value}" for key, value in seq_record.attributes.items()])
            fasta_file.write(f">{seq_record.seq_id} {attributes_str}\n{seq_record.seq}\n")
            n_written += 1
    return n_written
