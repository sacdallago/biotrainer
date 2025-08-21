import re

from pathlib import Path
from typing import Union, List

from ..protocols import Protocol
from ..utilities import RESIDUE_TO_VALUE_TARGET_DELIMITER
from ..input_files import BiotrainerSequenceRecord, read_FASTA


class InputValidator:

    def __init__(self, protocol: Protocol):
        self.protocol = protocol

    def validate(self, input_data: Union[str, Path, List[BiotrainerSequenceRecord]]) -> List[BiotrainerSequenceRecord]:
        if isinstance(input_data, str) or isinstance(input_data, Path):
            input_data = read_FASTA(input_data)

        if len(input_data) == 0:
            raise ValueError("The input to validate is empty!")

        if not isinstance(input_data[0], BiotrainerSequenceRecord):
            raise ValueError(f"Expected BiotrainerSequenceRecord, got {type(input_data[0])}!")

        for error in (self._validate_unique_sequences(input_data),
                      self._validate_sequences(input_data),
                      self._validate_targets(input_data)):
            if error != "":
                raise ValueError(error)

        return input_data

    @staticmethod
    def _validate_unique_sequences(input_data: List[BiotrainerSequenceRecord]) -> str:
        len_data = len(input_data)
        unique_sequence_data = {seq_record.seq: seq_record for seq_record in input_data}
        len_unique = len(unique_sequence_data)
        if len_data != len_unique:
            affected_seqs = [seq_record.seq for seq_record in input_data
                            if seq_record not in unique_sequence_data.values()]
            return (f"There are {len_data - len_unique} duplicated sequences in the input file!\n"
                    f"Affected sequences: {affected_seqs}")
        return ""

    @staticmethod
    def _validate_sequences(input_data: List[BiotrainerSequenceRecord]) -> str:
        # Regular expression pattern that matches any character that is not a letter
        invalid_char_pattern = re.compile(r'[^a-zA-Z]')

        for seq_record in input_data:
            # Check if there are any invalid characters in the sequence
            invalid_match = invalid_char_pattern.search(seq_record.seq)
            if invalid_match:
                return f"Invalid character '{invalid_match.group()}' found in sequence {seq_record.seq_id}!"

        return ""

    def _validate_targets(self, input_data: List[BiotrainerSequenceRecord]) -> str:
        # Expect float or int for regression
        for seq_record in input_data:
            target = seq_record.get_target()
            if target is None and seq_record.get_set().lower() != "pred":
                return f"No target found for sequence {seq_record.seq_id}!"
            if self.protocol in Protocol.regression_protocols():
                try:
                    targets = [target]

                    # r2v
                    if RESIDUE_TO_VALUE_TARGET_DELIMITER in target:
                        if self.protocol != Protocol.residue_to_value:
                            return (f"Found {RESIDUE_TO_VALUE_TARGET_DELIMITER} in {target} for "
                                    f"sequence {seq_record.seq_id} - "
                                    f"but protocol is not {Protocol.residue_to_value.name}!")
                        targets = target.split(RESIDUE_TO_VALUE_TARGET_DELIMITER)

                    for target in targets:
                        float(target)

                except ValueError:
                    return f"Invalid regression target {target} for sequence {seq_record.seq_id}!"

        return ""
