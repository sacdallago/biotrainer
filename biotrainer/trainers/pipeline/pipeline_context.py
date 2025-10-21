from pathlib import Path
from typing import Dict, Any, Union, List, Optional

from ...output_files import OutputManager
from ..target_manager import TargetManager
from ...embedders import PeftEmbeddingService
from ...input_files import BiotrainerSequenceRecord


class PipelineContext:
    """Context object that maintains state throughout the pipeline execution"""

    def __init__(self, config: Dict[str, Any], output_manager: OutputManager, custom_pipeline: bool):
        # Values set prior to pipeline execution
        self.config = config
        self.output_manager = output_manager
        self.custom_pipeline = custom_pipeline

        # Data produced during pipeline execution
        # Setup
        self.pipeline_start_time = None
        self.model_hash = None
        self.hp_manager = None
        # Input Data
        self.input_data: Optional[Union[Path, List[BiotrainerSequenceRecord]]] = None
        # Embedding + Projection
        self.id2emb = None
        self.embedding_service: Optional[PeftEmbeddingService] = None  # For fine-tuning only
        # Data Loading
        self.target_manager: Optional[TargetManager] = None
        self.n_features = None
        self.n_classes = None
        self.class_str2int: Optional[Dict[str, int]] = None  # Used to apply random_masking
        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets = None
        self.baseline_test_datasets = None  # For random model baseline, uses non-finetuned embeddings
        self.prediction_dataset = None
        self.class_weights = None

        # Training
        self.best_split = None

        # Timing information
        self.pipeline_end_time = None
        self.step_timings = {}
