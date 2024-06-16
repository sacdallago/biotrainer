from __future__ import annotations

from enum import Enum
from typing import List


class Protocol(Enum):
    residue_to_class = 1,
    residues_to_class = 2,
    residues_to_value = 3,
    sequence_to_class = 4,
    sequence_to_value = 5

    @staticmethod
    def all() -> List[Protocol]:
        return [Protocol.residue_to_class,
                Protocol.residues_to_class,
                Protocol.residues_to_value,
                Protocol.sequence_to_class,
                Protocol.sequence_to_value]

    @staticmethod
    def classification_protocols() -> List[Protocol]:
        return [Protocol.residue_to_class, Protocol.residues_to_class, Protocol.sequence_to_class]

    @staticmethod
    def regression_protocols() -> List[Protocol]:
        return [Protocol.residues_to_value, Protocol.sequence_to_value]

    @staticmethod
    def per_sequence_protocols() -> List[Protocol]:
        return [Protocol.residues_to_class, Protocol.residues_to_value,
                Protocol.sequence_to_class, Protocol.sequence_to_value]

    @staticmethod
    def per_residue_protocols() -> List[Protocol]:
        return [Protocol.residue_to_class]

    @staticmethod
    def from_string(string: str) -> Protocol:
        return {p.name: p for p in Protocol.all()}[string]

    def __str__(self):
        return self.name
