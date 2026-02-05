from __future__ import annotations

from .constants import RESIDUE_TO_VALUE_TARGET_DELIMITER

from ..protocols import Protocol

from collections import namedtuple
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict, Union

Split = namedtuple("Split", "name train val")
SplitResult = namedtuple("SplitResult", "name, hyper_params, best_epoch_metrics, solver")
EmbeddingDatasetSample = namedtuple("EmbeddingDatasetSample", "seq_id embedding target")
SequenceDatasetSample = namedtuple("SequenceDatasetSample", "seq_id, seq_record target")


class EpochMetrics(BaseModel):
    epoch: int = Field(description="Epoch number")
    training: Dict = Field(description="Training metrics")
    validation: Dict = Field(description="Validation metrics")

    def to_dict(self):
        return {"epoch": self.epoch, "training": self.training, "validation": self.validation}


class MetricEstimate(BaseModel):
    name: str = Field(description="Name of the metric")
    mean: float = Field(description="Mean of the metric values")
    lower: float = Field(description="Lower bound of the metric values")
    upper: float = Field(description="Upper bound of the metric values")


class BootstrappedMetric(MetricEstimate):
    iterations: int = Field(description="Number of iterations used for bootstrapping")
    sample_size: int = Field(description="Sample size used for bootstrapping")
    confidence_level: float = Field(description="Confidence level used for bootstrapping")


class BiotrainerSequencePrediction(BaseModel):
    seq_id: str = Field(description="Sequence identifier")
    prediction: Any = Field(description="Predicted value")
    mcd_predictions: Optional[List[Any]] = Field(default=None, description="All Monte-Carlo-Dropout predictions")
    mcd_mean: Optional[Union[float, List[float]]] = Field(default=None, description="Monte-Carlo-Dropout mean(s)")
    mcd_std: Optional[Union[float, List[float]]] = Field(default=None,
                                                         description="Monte-Carlo-Dropout standard deviation(s)")
    mcd_lower_bound: Optional[Union[float, List[float]]] = Field(default=None,
                                                                 description="Monte-Carlo-Dropout lower bound(s)")
    mcd_upper_bound: Optional[Union[float, List[float]]] = Field(default=None,
                                                                 description="Monte-Carlo-Dropout upper bound(s)")
    bald_score: Optional[float] = Field(default=None, description="BALD score")

    def revert_mappings(self, protocol: Protocol,
                        class_int2str: Optional[Dict[int, str]] = None) -> BiotrainerSequencePrediction:
        if class_int2str is None:
            return self

        pred = self.prediction
        mcd_preds = self.mcd_predictions  # No remapping for per-residue at the moment
        if protocol == Protocol.residue_to_class:
            pred = [class_int2str[int(p)] for p in pred]
        elif protocol in Protocol.classification_protocols():
            pred = class_int2str[int(pred)]
            mcd_preds = [class_int2str[int(mcd_pred)] for mcd_pred in mcd_preds]
        return BiotrainerSequencePrediction(seq_id=self.seq_id, prediction=pred, mcd_predictions=mcd_preds,
                                            mcd_mean=self.mcd_mean, mcd_std=self.mcd_std,
                                            mcd_lower_bound=self.mcd_lower_bound,
                                            mcd_upper_bound=self.mcd_upper_bound,
                                            bald_score=self.bald_score
                                            )


class BiotrainerResiduePrediction(BiotrainerSequencePrediction):
    residue_index: int = Field(description="Residue index")

    @staticmethod
    def collapse_predictions(predictions: List[BiotrainerResiduePrediction],
                             protocol: Protocol) -> List[BiotrainerSequencePrediction]:
        """ Collapse predictions into a single prediction for each sequence"""
        seq_preds = {}
        for pred in predictions:
            if pred.seq_id not in seq_preds:
                seq_preds[pred.seq_id] = []
            seq_preds[pred.seq_id].append(pred)

        delimiter = RESIDUE_TO_VALUE_TARGET_DELIMITER if protocol in Protocol.regression_protocols() else ""
        return [BiotrainerSequencePrediction(seq_id=seq_id,
                                             prediction=delimiter.join([str(pred.prediction) for pred in
                                                                        sorted(preds, key=lambda p: p.residue_index)]))
                for seq_id, preds in seq_preds.items()]
