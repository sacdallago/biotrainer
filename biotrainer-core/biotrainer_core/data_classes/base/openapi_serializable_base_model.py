from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Dict, Optional, Self

class OpenAPISerializableBaseModel(BaseModel):
    """Base class providing OpenAPI client serialization compatibility for Pydantic models."""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_dict(cls: type[Self], obj: Any) -> Optional[Self]:
        if obj is None:
            return None
        if isinstance(obj, cls):
            return obj
        return cls.model_validate(obj)