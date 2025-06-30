from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Type, Any, List, Tuple, Callable, Dict

from ..protocols import Protocol


@dataclass
class ConfigOption:
    name: str
    required: bool
    description: str
    category: str
    is_file_option: bool = False
    allow_hyperparameter_optimization: bool = False
    add_if: Optional[Callable[[Dict[str, Any]], bool]] = None
    default: Optional[Any] = None
    constraints: Optional[ConfigConstraints] = None

    def to_dict(self):
        return {
            "name": self.name,
            "required": self.required,
            "description": self.description,
            "category": self.category,
            "is_file_option": self.is_file_option,
            "allow_hyperparameter_optimization": self.allow_hyperparameter_optimization,
            "default": self.default,
            "constraints": self.constraints.to_dict() if self.constraints else {},
        }

@dataclass
class ConfigConstraints:
    type: Optional[Type] = None
    allowed_values: Optional[List[Any]] = None
    allowed_formats: Optional[List[str]] = None
    allowed_protocols: Optional[List[Protocol]] = field(default_factory=lambda: Protocol.all())
    custom_validator: Optional[Callable[[Any], Tuple[bool, str]]] = None  # Function that returns validation and error
    lt: Optional[int] = None
    lte: Optional[int] = None
    gt: Optional[int] = None
    gte: Optional[int] = None

    def to_dict(self):
        return {
            "type": str(self.type.__name__),
            "allowed_values": self.allowed_values,
            "allowed_formats": self.allowed_formats,
            "allowed_protocols": [protocol.name for protocol in self.allowed_protocols],
            "lt": self.lt,
            "lte": self.lte,
            "gt": self.gt,
            "gte": self.gte,
        }

class ConfigKey(Enum):
    ROOT = ""
    HF_DATASET = "hf_dataset"
    CROSS_VALIDATION = "cross_validation_config"
    FINETUNING = "finetuning_config"

    @staticmethod
    def all_subconfig_keys() -> List[ConfigKey]:
        return [key for key in ConfigKey if key != ConfigKey.ROOT]
