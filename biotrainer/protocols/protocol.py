from __future__ import annotations

from enum import Enum
from typing import List


class Protocol(Enum):
    residue_to_class = 1,
    residues_to_class = 2,
    sequence_to_class = 3,
    sequence_to_value = 4

    @staticmethod
    def all() -> List[Protocol]:
        return [Protocol.residue_to_class,
                Protocol.residues_to_class,
                Protocol.sequence_to_class,
                Protocol.sequence_to_value]

    @staticmethod
    def regression_protocols() -> List[Protocol]:
        return [Protocol.sequence_to_value]

    @staticmethod
    def per_protein_protocols() -> List[Protocol]:
        return [Protocol.residues_to_class, Protocol.sequence_to_class, Protocol.sequence_to_value]

    @staticmethod
    def per_residue_protocols() -> List[Protocol]:
        return [Protocol.residue_to_class]

    def __str__(self):
        return self.name
