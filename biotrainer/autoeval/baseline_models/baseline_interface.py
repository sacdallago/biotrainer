from abc import abstractmethod, ABC
from ...bioengineer import BioEngineer
from ...embedding import CustomEmbedder

class AutoEvalBaseline(ABC):
    @abstractmethod
    def embedder(self) -> CustomEmbedder:
        raise NotImplementedError

    @abstractmethod
    def bioengineer(self) -> BioEngineer:
        raise NotImplementedError
