from __future__ import annotations
import numpy as np

from typing import Dict, Any, Union, Optional, List

from ..utilities import calculate_sequence_hash


class BiotrainerSequenceRecord:
    def __init__(self, seq_id: str, seq: str, attributes: Optional[Dict[str, Any]] = None,
                 embedding: Optional[np.ndarray] = None):
        self.seq_id = seq_id
        self.seq = seq
        self.attributes = {k.upper(): v for k, v in attributes.items()} if attributes else {}
        self.embedding = embedding

    def get_target(self) -> Union[None, str, float]:
        return self.attributes.get("TARGET")

    def get_mask(self) -> Union[None, str]:
        return self.attributes.get("MASK")

    def get_set(self) -> Union[None, str]:
        return self.attributes.get("SET")

    def get_deprecated_set(self) -> Union[None, str]:
        if "SET" not in self.attributes:
            return None
        set_name = self.attributes["SET"]
        if set_name.lower() == "train":
            val = self.attributes.get("VALIDATION")
            if val is not None:
                val = eval(val)
                set_name = "val" if val else "train"
        return set_name

    def get_ppi(self) -> Union[None, str]:
        """ Get the INTERACTOR id (i.e. another sequence id in the same fasta file) """
        return self.attributes.get("INTERACTOR")

    def get_hash(self) -> str:
        return calculate_sequence_hash(self.seq)

    def copy_with_embedding(self, embedding: np.ndarray) -> BiotrainerSequenceRecord:
        """ Set the embedding for this sequence record and return sequence record """
        return BiotrainerSequenceRecord(seq_id=self.seq_id, seq=self.seq,
                                        attributes=self.attributes, embedding=embedding)

    @staticmethod
    def get_dicts(input_records: List[BiotrainerSequenceRecord]) -> (dict, dict, dict):
        id2targets = {}
        id2masks = {}
        id2sets = {}
        for seq_record in input_records:
            seq_hash = seq_record.get_hash()
            id2targets[seq_hash] = seq_record.get_target()
            mask = seq_record.get_mask()
            id2masks[seq_hash] = np.array([int(mask_value) for mask_value in mask]) if mask else None
            id2sets[seq_hash] = seq_record.get_set()
        return id2targets, id2masks, id2sets
