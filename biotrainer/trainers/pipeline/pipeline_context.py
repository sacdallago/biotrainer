from typing import Dict, Any

from ...output_files import OutputManager


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
        # Embedding + Projection
        self.id2emb = None
        # Data Loading
        self.target_manager = None
        self.n_features = None
        self.n_classes = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets = None
        self.prediction_dataset = None
        self.class_weights = None

        #self.splits = []
        #self.split_results = []
        # self.pipeline_metrics = {}

        # Training
        self.best_split = None

        # Timing information
        self.pipeline_end_time = None
        self.step_timings = {}
