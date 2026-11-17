from __future__ import annotations

from ruamel import yaml
from pathlib import Path
from pydantic import Field
from typing import Any, Dict, Optional, List, Union, Tuple

from .embedding_stats import EmbeddingStats
from .base import OpenAPISerializableBaseModel
from .metrics import EpochMetrics, BootstrappedMetric
from .biotrainer_prediction import BiotrainerPrediction, BiotrainerInferenceResult


class DerivedValues(OpenAPISerializableBaseModel):
    """ Derived values calculated during the training process. """
    biotrainer_version: Optional[str] = Field(default=None, description="Version of BioTrainer used for training")
    class_int2str: Optional[Dict[int, str]] = Field(default=None,
                                                    description="Mapping of class integers to class names")
    class_str2int: Optional[Dict[str, int]] = Field(default=None,
                                                    description="Mapping of class names to class integers")
    computed_class_weights: Optional[Dict[int, float]] = Field(default=None,
                                                               description="Class weights computed during training")
    embedding_stats: Optional[EmbeddingStats] = Field(default=None, description="Statistics of the embeddings")
    embeddings_file: Optional[str] = Field(default=None, description="Path to the embeddings file")
    model_hash: Optional[str] = Field(default=None, description="Hash of the model")
    n_classes: Optional[int] = Field(default=None, description="Number of classes in the dataset")
    n_features: Optional[int] = Field(default=None, description="Number of input features (e.g. embedding dimensions)")
    n_testing_ids: Optional[int] = Field(default=None, description="Number of sequences in the test set")
    pipeline_elapsed_time: Optional[float] = Field(default=None, description="Elapsed time in seconds for the pipeline")
    pipeline_end_time: Optional[str] = Field(default=None, description="End time of the pipeline")
    pipeline_start_time: Optional[str] = Field(default=None, description="Start time of the pipeline")
    training_elapsed_time: Optional[float] = Field(default=None, description="Elapsed time in seconds for training")


class TrainingResult(OpenAPISerializableBaseModel):
    """ Training results for each cross-validation split. """
    n_training_ids: Optional[int] = Field(default=None, description="Number of sequences in the training set")
    n_validation_ids: Optional[int] = Field(default=None, description="Number of sequences in the validation set")
    training_ids: Optional[List[str]] = Field(default=None, description="List of IDs in the training set")
    validation_ids: Optional[List[str]] = Field(default=None, description="List of IDs in the validation set")
    split_hyper_params: Optional[Dict[str, Any]] = Field(default=None,
                                                         description="Hyperparameters used for this split")
    n_free_parameters: Optional[int] = Field(default=None, description="Number of free parameters in the model")
    start_time: Optional[str] = Field(default=None, description="Start time of the training process")
    end_time: Optional[str] = Field(default=None, description="End time of the training process")
    elapsed_time: Optional[float] = Field(default=None, description="Elapsed time in seconds for training")
    training_losses: List[float] = Field(default_factory=list, description="Training losses for each epoch")
    validation_losses: List[float] = Field(default_factory=list, description="Validation losses for each epoch")
    best_epoch_metrics: Optional[EpochMetrics] = Field(default=None, description="Best training epoch metrics")
    sanity_check_warnings: Optional[List[str]] = Field(default=None, description="Warnings from sanity checks")


class TestResult(OpenAPISerializableBaseModel):
    """ Test results after training. """
    inference_result: Optional[BiotrainerInferenceResult] = Field(default=None,
                                                                  description="Plain test inference result")
    bootstrapped_metrics: Optional[List[BootstrappedMetric]] = Field(default=None,
                                                                     description="Bootstrapped test metrics")
    baselines: Optional[Dict[str, List[BootstrappedMetric]]] = Field(default=None,
                                                                     description="Bootstrapped baselines by method name")
    sanity_check_warnings: Optional[List[str]] = Field(default=None, description="Warnings from sanity checks")


class BiotrainerModelResult(OpenAPISerializableBaseModel):
    config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration parameters")
    derived_values: Optional[DerivedValues] = Field(default=None,
                                                    description="Values derived during the training process")
    training_results: Dict[str, TrainingResult] = Field(default_factory=dict,
                                                        description="Training results for each cross-validation split")
    test_results: Dict[str, TestResult] = Field(default_factory=dict,
                                                description="Test results after training for each test set")
    predictions: List[BiotrainerPrediction] = Field(default_factory=list, description="Predictions made by the model")

    @classmethod
    def from_file(cls, file_path: Union[Path, str]):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix in [".yml", ".yaml"]:
            with open(file_path, "r") as tr_file:
                training_output = yaml.load(tr_file, Loader=yaml.RoundTripLoader)
            return cls.model_validate(training_output, strict=False)
        elif file_path.suffix in [".json"]:
            training_output = file_path.read_text()
            return cls.model_validate_json(training_output)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")


class BiotrainerModelUpdate(OpenAPISerializableBaseModel):
    current_model_result: BiotrainerModelResult = Field(description="Current model result")
    training_iteration: Optional[Tuple[str, EpochMetrics]] = Field(default=None,
                                                                   description="Current training iteration for fast "
                                                                               "updates of observers like tensorboard")
