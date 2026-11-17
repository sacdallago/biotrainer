from __future__ import annotations

import ast
import hashlib

from typing import Dict, Any, Union, Optional, List, Tuple, Iterable
from pydantic import Field, model_validator, field_serializer

from .base import OpenAPISerializableBaseModel

from ..functions.hashing import calculate_sequence_hash
from ..utils.constants import RESIDUE_TO_VALUE_TARGET_DELIMITER

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False


class SequenceData(OpenAPISerializableBaseModel):
    seq_id: str = Field(description="Sequence id", min_length=1)
    seq: str = Field(description="Sequence")

    # User-friendly named parameters (mapped into attributes)
    label: Optional[str] = Field(default=None, description="Shortcut for TARGET attribute")
    set: Optional[str] = Field(default=None, description="Shortcut for SET attribute")
    mask: Optional[str] = Field(default=None, description="Shortcut for MASK attribute")

    # Generic attributes dict (stores everything)
    attributes: Optional[Dict[str, Any]] = Field(default=None, description="Attributes such as TARGET, SET or MASK")
    embedding: Optional[Union[list, Any]] = Field(default=None, description="Embedding (should be a list or torch.tensor or numpy array)")

    @field_serializer('embedding')
    def serialize_embedding(self, embedding: Any, _info):
        # This ensures the field is never serialized to JSON to save space,
        # but keeps the field in the generated schema for openapi.
        return None

    @model_validator(mode="after")
    def validate_record(self):
        # Normalize attribute keys to uppercase
        attrs = {k.upper(): v for k, v in (self.attributes or {}).items()}

        # Validate conflicts between named parameters and attributes dict
        if self.label is not None and "TARGET" in attrs:
            if self.label != attrs["TARGET"]:
                raise ValueError(
                    f"Conflict: 'label' parameter ('{self.label}') differs from "
                    f"'TARGET' in attributes ('{attrs['TARGET']}'). Use one or the other."
                )
        if self.set is not None and "SET" in attrs:
            if self.set != attrs["SET"]:
                raise ValueError(
                    f"Conflict: 'set' parameter ('{self.set}') differs from "
                    f"'SET' in attributes ('{attrs['SET']}'). Use one or the other."
                )
        if self.mask is not None and "MASK" in attrs:
            if self.mask != attrs["MASK"]:
                raise ValueError(
                    f"Conflict: 'mask' parameter ('{self.mask}') differs from "
                    f"'MASK' in attributes ('{attrs['MASK']}'). Use one or the other."
                )

        # Merge named parameters into attributes (named params take precedence)
        if self.label is not None:
            attrs["TARGET"] = self.label
        if self.set is not None:
            attrs["SET"] = self.set
        if self.mask is not None:
            attrs["MASK"] = self.mask

        self.attributes = attrs

        # Reallocate vice versa to ensure consistency
        self.label = self.attributes.get("TARGET", self.label)
        self.set = self.attributes.get("SET", self.set)
        self.mask = self.attributes.get("MASK", self.mask)

        # Validate mask length matches sequence length
        mask_val = self.attributes.get("MASK")
        if mask_val is not None and len(mask_val) != len(self.seq):
            raise ValueError(
                f"Length of MASK ({len(mask_val)}) must match length of sequence ({len(self.seq)})"
            )

        return self

    # --- Generic attribute access ---

    def get_attribute(self, key: str) -> Optional[Any]:
        """Get any attribute by key (case-insensitive)."""
        return self.attributes.get(key.upper())

    def set_attribute(self, key: str, value: Any) -> SequenceData:
        """Return a copy with the given attribute set."""
        old_attrs = self.attributes or dict()
        new_attrs = dict(old_attrs)
        new_attrs[key.upper()] = value
        return SequenceData(
            seq_id=self.seq_id, seq=self.seq,
            attributes=new_attrs, embedding=self.embedding
        )

    # --- Training-specific convenience accessors ---

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
                val = ast.literal_eval(val)
                set_name = "val" if val else "train"
        return set_name

    def get_ppi(self) -> Union[None, str]:
        """ Get the INTERACTOR id (i.e. another sequence id in the same fasta file) """
        return self.attributes.get("INTERACTOR")

    def get_torch_embedding(self):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")
        embd = self.embedding
        if isinstance(embd, torch.Tensor):
            return embd.float()
        if _NUMPY_AVAILABLE:
            if isinstance(embd, np.ndarray):
                return torch.tensor(embd, dtype=torch.float)
        if isinstance(embd, list) or isinstance(embd, Iterable):
            return torch.tensor(embd, dtype=torch.float)
        raise ValueError(f"Invalid embedding type: {type(embd)}")

    # --- Hash / ID ---

    def get_hash(self) -> str:
        return calculate_sequence_hash(self.seq)

    def get_id_for_id2emb(self):
        if self.seq is not None and self.seq != "":
            return self.get_hash()
        return self.seq_id

    def get_embedding_hash(self) -> Optional[str]:
        """
        Efficiently hash the embedding via its raw byte representation.
        Returns None if no embedding is set.

        NOTE: We normalize to float64 before hashing to avoid false cache
        misses caused purely by dtype differences (e.g. float32 vs float64)
        when the underlying values are identical.
        """
        if self.embedding is None:
            return None

        if _TORCH_AVAILABLE and isinstance(self.embedding, torch.Tensor):
            raw_bytes = self.embedding.cpu().detach().double().numpy().tobytes()
        elif _NUMPY_AVAILABLE and isinstance(self.embedding, np.ndarray):
            raw_bytes = self.embedding.astype(np.float64).tobytes()
        elif isinstance(self.embedding, list):
            if _NUMPY_AVAILABLE:
                raw_bytes = np.array(self.embedding, dtype=np.float64).tobytes()
            else:
                # Fallback: repr is slow but only hits when numpy is absent
                raw_bytes = repr(self.embedding).encode()
        else:
            raw_bytes = repr(self.embedding).encode()

        return hashlib.sha256(raw_bytes).hexdigest()

    # --- Copy helpers ---

    def copy_with_embedding(self, embedding) -> SequenceData:
        """ Set the embedding for this sequence record and return sequence record """
        return self.model_copy(update={"embedding": embedding}, deep=True)

    def copy_without_label(self) -> SequenceData:
        """Remove label and set to 'pred' (for active learning simulations)."""
        new_attrs = {k: v for k, v in self.attributes.items() if k != "TARGET"}
        new_attrs["SET"] = "pred"
        return SequenceData(
            seq_id=self.seq_id, seq=self.seq,
            attributes=new_attrs, embedding=self.embedding
        )

    def copy_with_label(self, label: str, set_name: str = "train") -> SequenceData:
        """Set label and set (e.g. for active learning simulations)."""
        old_attrs = self.attributes or dict()
        new_attrs = dict(old_attrs)
        new_attrs["TARGET"] = label
        new_attrs["SET"] = set_name
        return SequenceData(
            seq_id=self.seq_id, seq=self.seq,
            attributes=new_attrs, embedding=self.embedding
        )

    # --- Serialization ---

    def to_fasta(self) -> str:
        """Serialize to FASTA format with all attributes in the header."""
        header_parts = [f">{self.seq_id}"]
        for key, value in self.attributes.items():
            if value is not None:
                header_parts.append(f"{key}={value}")
        return " ".join(header_parts) + "\n" + self.seq

    def to_dict(self) -> Dict[str, Any]:
        """Return the dictionary representation of the model using alias.

        This has the following differences from calling pydantic's
        `self.model_dump(by_alias=True)`:

        * `None` is only added to the output dict for nullable fields that
          were set at model initialization. Other fields with value `None`
          are ignored.
        """
        excluded_fields = set([])

        _dict = self.model_dump(
            by_alias=True,
            exclude=excluded_fields,
            exclude_none=True,
        )
        # set to None if label (nullable) is None
        # and model_fields_set contains the field
        if self.label is None and "label" in self.model_fields_set:
            _dict['label'] = None

        # set to None if set (nullable) is None
        # and model_fields_set contains the field
        if self.set is None and "set" in self.model_fields_set:
            _dict['set'] = None

        # set to None if mask (nullable) is None
        # and model_fields_set contains the field
        if self.mask is None and "mask" in self.model_fields_set:
            _dict['mask'] = None

        # set to None if attributes (nullable) is None
        # and model_fields_set contains the field
        if self.attributes is None and "attributes" in self.model_fields_set:
            _dict['attributes'] = None

        # set to None if embedding (nullable) is None
        # and model_fields_set contains the field
        if self.embedding is None and "embedding" in self.model_fields_set:
            _dict['embedding'] = None

        return _dict

    # --- Static helpers ---

    @staticmethod
    def _convert_regression_target_if_necessary(target: Optional):
        if target is None:  # Can be the case for the predict dataset
            return target
        if RESIDUE_TO_VALUE_TARGET_DELIMITER in str(target):
            targets = target.split(RESIDUE_TO_VALUE_TARGET_DELIMITER)
            return list(map(float, targets))
        return target

    @staticmethod
    def get_dicts(input_records: List[SequenceData]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        id2targets = {}
        id2masks = {}
        id2sets = {}
        for seq_record in input_records:
            seq_hash = seq_record.get_id_for_id2emb()

            target = seq_record.get_target()
            target = SequenceData._convert_regression_target_if_necessary(target)
            id2targets[seq_hash] = target

            mask = seq_record.get_mask()
            if _NUMPY_AVAILABLE:
                id2masks[seq_hash] = np.array([int(mask_value) for mask_value in mask]) if mask else None
            else:
                id2masks[seq_hash] = [int(mask_value) for mask_value in mask] if mask else None

            id2sets[seq_hash] = seq_record.get_set()
        return id2targets, id2masks, id2sets
