from __future__ import annotations

from typing import Optional
from abc import ABC, abstractmethod

from .autoeval_config_bank import AutoEvalConfigBank
from .autoeval_data_handler import AutoEvalDataHandler

from biotrainer_core.data_classes.autoeval import AutoEvalMode


class AutoEvalFramework(ABC):

    def __init__(self):
        self._data_handler = self.make_data_handler()
        self._config_bank = self.make_config_bank()

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
    def make_data_handler(self) -> AutoEvalDataHandler:
        raise NotImplementedError

    @abstractmethod
    def make_config_bank(self) -> AutoEvalConfigBank:
        raise NotImplementedError

    def get_data_handler(self) -> AutoEvalDataHandler:
        return self._data_handler

    def get_config_bank(self) -> AutoEvalConfigBank:
        return self._config_bank