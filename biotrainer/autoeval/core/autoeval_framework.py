from __future__ import annotations

from enum import Enum
from typing import Optional
from abc import ABC, abstractmethod

from .autoeval_config_bank import AutoEvalConfigBank
from .autoeval_data_handler import AutoEvalDataHandler

class AutoEvalMode(Enum):
    SUPERVISED = "SUPERVISED"
    ZERO_SHOT = "ZERO_SHOT"


class AutoEvalFramework(ABC):
    @classmethod
    def detect(cls, framework_name: str) -> Optional[AutoEvalFramework]:
        if framework_name.lower() == cls.get_name().lower():
            return cls()
        return None

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_mode() -> AutoEvalMode:
        raise NotImplementedError

    @abstractmethod
    def get_data_handler(self) -> AutoEvalDataHandler:
        raise NotImplementedError

    @abstractmethod
    def get_config_bank(self) -> AutoEvalConfigBank:
        raise NotImplementedError