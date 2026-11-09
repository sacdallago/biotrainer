import torch

from pathlib import Path
from biotrainer.shared import get_device
from typing import Optional, Union, Dict, Any, List, Generator
from biotrainer_core.data_classes import SequenceData, Protocol

from .stats import EmbeddingStatsTracker
from .services import get_embedding_service


class EmbeddingAPI:
    """ High-level API for computing embeddings"""

    def __init__(self, embedder_name: str,
                 custom_tokenizer_config: Optional[str] = None,
                 use_half_precision: Optional[bool] = False,
                 device: Optional[Union[str, torch.device]] = None,
                 finetuning_config: Optional[Dict[str, Any]] = None):
        self._embedder_name = embedder_name
        self._custom_tokenizer_config = custom_tokenizer_config
        self._use_half_precision = use_half_precision
        self._device = device
        self._finetuning_config = finetuning_config

        self._embedding_service = get_embedding_service(embedder_name, custom_tokenizer_config, use_half_precision,
                                                        get_device(device), finetuning_config)

    def compute_embeddings(self, input_data: Union[str, Path, List[str], List[SequenceData], Dict[str, SequenceData]],
                           output_dir: Path,
                           protocol: Protocol,
                           force_output_dir: bool = False,
                           force_recomputing: bool = False,
                           store_by_hash: bool = True,
                           embedding_stats_tracker: EmbeddingStatsTracker = None) -> str:
        """ Computes embeddings and returns the output file path """
        return self._embedding_service.compute_embeddings(input_data, output_dir, protocol, force_output_dir,
                                                          force_recomputing, store_by_hash, embedding_stats_tracker)

    def generate_embeddings(self, input_data: Union[str, Path, List[str], List[SequenceData], Dict[
                                str, SequenceData]],
                            reduce: bool) -> Generator[SequenceData, None, None]:
        """ Generates embeddings and yields them as they are computed.
            The SequenceData objects are unified with the computed embedding
        """
        return self._embedding_service.generate_embeddings(input_data, reduce)