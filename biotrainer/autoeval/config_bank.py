from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional

from .data_handler import AutoEvalTask


class AutoEvalConfigBank(ABC):

    @abstractmethod
    def get_task_config(self, task: AutoEvalTask) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def add_custom_values_to_config(config: Dict[str, Any],
                                    embedder_name: str,
                                    input_file: Union[str, Path],
                                    output_dir: Union[str, Path],
                                    embeddings_file: Optional[Union[str, Path]] = None,
                                    ) -> Dict[str, Any]:
        config.update({"embedder_name": embedder_name,
                       "input_file": str(input_file),
                       "output_dir": str(output_dir)})
        if embeddings_file is not None:
            config.update({"embeddings_file": str(embeddings_file)})
        return config
