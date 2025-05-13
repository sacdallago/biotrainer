from __future__ import annotations
import numpy as np

from typing import Dict, Any, Union, List


class BiotrainerSequenceRecord:
    def __init__(self, seq_id: str, seq: str, attributes: Dict[str, Any]):
        self.seq_id = seq_id
        self.seq = seq
        self.attributes = {k.upper(): v for k, v in attributes.items()}

    def get_target(self) -> Union[None, str, float]:
        return self.attributes.get("TARGET")

    def get_mask(self) -> Union[None, str]:
        return self.attributes.get("MASK")

    def get_set(self) -> Union[None, str]:
        return self.attributes.get("SET")

    def get_ppi(self) -> Union[None, str]:
        """ Get the INTERACTOR id (i.e. another sequence id in the same fasta file) """
        return self.attributes.get("INTERACTOR")

    @staticmethod
    def get_dicts(input_records: Dict[str, BiotrainerSequenceRecord]) -> (dict, dict, dict):
        id2targets = {}
        id2masks = {}
        id2sets = {}
        for seq_id, seq_record in input_records.items():
            id2targets[seq_id] = seq_record.get_target()
            mask = seq_record.get_mask()
            id2masks[seq_id] = np.array([int(mask_value) for mask_value in mask]) if mask else None
            id2sets[seq_id] = seq_record.get_set()
        return id2targets, id2masks, id2sets
