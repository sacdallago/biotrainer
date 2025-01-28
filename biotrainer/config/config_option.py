from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type, Any, List, Tuple, Callable

from ..protocols import Protocol


@dataclass
class ConfigOption:
    name: str
    required: bool
    is_file_option: bool = False
    allow_hyperparameter_optimization: bool = False
    type: Optional[Type] = None
    default: Optional[Any] = None
    constraints: Optional[ConfigConstraints] = None
    # TODO description
    # TODO category


@dataclass
class ConfigConstraints:
    allowed_values: Optional[List[Any]] = None
    allowed_formats: Optional[List[str]] = None
    allowed_protocols: Optional[List[Protocol]] = field(default_factory=lambda: Protocol.all())
    custom_validator: Optional[Callable[[Any], Tuple[bool, str]]] = None  # Function that returns validation and error
    lt: Optional[int] = None
    lte: Optional[int] = None
    gt: Optional[int] = None
    gte: Optional[int] = None
